
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load, dump
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

st.set_page_config(page_title='Insurance Fraud Detection (Isolation Forest)', layout='wide')
st.title('🔎 Insurance Fraud Detection — Streamlit Cloud App')

st.markdown(
    'Train an **Isolation Forest** or **score** uploaded claims. '
    'Includes light feature engineering and works with numeric + categorical columns.'
)

# ------------------------------
# Feature Engineering / Prep
# ------------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ratios (robust to missing values)
    if {'claim_amount', 'premium'}.issubset(df.columns):
        df['claim_to_premium_ratio'] = (df['claim_amount'].astype(float) + 1.0) / (df['premium'].astype(float) + 1.0)
    if {'claim_amount', 'sum_insured'}.issubset(df.columns):
        df['claim_to_suminsured_ratio'] = (df['claim_amount'].astype(float) + 1.0) / (df['sum_insured'].astype(float) + 1.0)

    # Dates
    if 'claim_date' in df.columns:
        dtcol = pd.to_datetime(df['claim_date'], errors='coerce')
        df['day_of_week'] = dtcol.dt.dayofweek
        df['month'] = dtcol.dt.month

    # Crop rain mismatch heuristic
    if {'product','event_type','rainfall_mm'}.issubset(df.columns):
        df['crop_rain_mismatch'] = ((df['product']=='crop') & (df['event_type']=='excess_rain') & (df['rainfall_mm'].astype(float) < 20)).astype(int)

    # Odd hour risk
    if 'claim_hour' in df.columns and 'claim_to_suminsured_ratio' in df.columns:
        df['odd_hour_risk'] = (df['claim_hour'].astype(float).isin([2,3,4]) & (df['claim_to_suminsured_ratio'] > 0.7)).astype(int)

    # Clean infs
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df

def build_preprocessor(df: pd.DataFrame):
    numeric_features = [
        "claim_hour","reported_latency_days","sum_insured","premium",
        "claim_amount","tenure_months","rainfall_mm","customer_num_claims",
        "customer_avg_claim_amount","customer_total_claim_amount",
        "claim_to_premium_ratio","claim_to_suminsured_ratio",
        "day_of_week","month","crop_rain_mismatch","odd_hour_risk"
    ]
    categorical_features = ["product","channel","region","event_type"]

    # Only keep columns that exist
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  PipelineCompat([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat",  PipelineCompat([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, categorical_features

class PipelineCompat:
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {}

    def fit(self, X, y=None):
        Xt = X
        for name, step in self.steps:
            step.fit(Xt, y)
            self.named_steps[name] = step
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
        self._Xt_ = Xt
        return self

    def transform(self, X):
        Xt = X
        for name, step in self.steps:
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# ------------------------------
# Utilities
# ------------------------------
def compute_metrics(y_true: pd.Series, scores: np.ndarray):
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["roc_auc"] = None
    try:
        out["avg_precision"] = float(average_precision_score(y_true, scores))
    except Exception:
        out["avg_precision"] = None
    return out

def score_with_pipeline(pipe, df_raw: pd.DataFrame):
    df = feature_engineer(df_raw)
    # Use the same columns used in training if available
    try:
        Xp = pipe["prep"].transform(df)
        scores = -pipe["model"].score_samples(Xp)  # higher = more anomalous
    except Exception as e:
        raise RuntimeError(f"Preprocessing or scoring failed: {e}")
    return scores, df

# ------------------------------
# Sidebar
# ------------------------------
mode = st.sidebar.radio("Mode", ["Score Claims", "Train Model"], index=0)
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
default_model_path = model_dir / "isoforest.pkl"

# ------------------------------
# Score Mode
# ------------------------------
if mode == "Score Claims":
    st.subheader("Score uploaded claims with a trained model")
    st.write("Upload a trained `isoforest.pkl` or use one from the working directory.")

    model_bytes = st.file_uploader("Upload trained model (.pkl) — optional", type=["pkl"])
    if model_bytes is not None:
        # Load from uploaded bytes
        pipe = load(io.BytesIO(model_bytes.read()))
        st.success("Loaded model from uploaded file.")
    elif default_model_path.exists():
        pipe = load(default_model_path)
        st.success(f"Loaded model from {default_model_path}")
    else:
        pipe = None
        st.warning("No model found. Train one in the 'Train Model' tab or upload a .pkl.")

    uploaded = st.file_uploader("Upload claims CSV to score", type=["csv"])

    if pipe is not None and uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        try:
            scores, df_feat = score_with_pipeline(pipe, df_raw)
        except Exception as e:
            st.error(str(e))
            st.stop()

        out = df_raw.copy()
        out["anomaly_score"] = scores
        out_sorted = out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

        st.write("### Top suspected claims")
        top_k = st.number_input("Show top K", min_value=10, max_value=1000, value=100, step=10)
        st.dataframe(out_sorted.head(int(top_k)))

        st.download_button(
            "Download scored CSV",
            data=out_sorted.to_csv(index=False),
            file_name="scored_claims.csv"
        )

        # Optional metrics if truth exists
        if "is_fraud" in out_sorted.columns:
            y_true = out_sorted["is_fraud"].fillna(0).astype(int).values
            metrics = compute_metrics(y_true, out_sorted["anomaly_score"].values)
            st.write("### Optional Metrics (requires `is_fraud` column)")
            st.json(metrics)

        # Simple histogram of scores (matplotlib)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.hist(out_sorted["anomaly_score"].values, bins=50)
        plt.title("Distribution of anomaly scores")
        plt.xlabel("anomaly_score")
        plt.ylabel("count")
        st.pyplot(fig)

# ------------------------------
# Train Mode
# ------------------------------
if mode == "Train Model":
    st.subheader("Train a new Isolation Forest model")
    st.caption("Note: In Streamlit Cloud, files written during a session are not persisted across restarts.")
    train_csv = st.file_uploader("Upload training claims CSV", type=["csv"], key="train_csv")
    contamination = st.slider("Contamination (expected anomaly fraction)", 0.001, 0.20, 0.05, 0.001)
    n_estimators = st.slider("n_estimators", 50, 400, 250, 25)

    if train_csv is not None:
        df_train_raw = pd.read_csv(train_csv)
        df_train = feature_engineer(df_train_raw)

        preprocessor, num_cols, cat_cols = build_preprocessor(df_train)

        model = IsolationForest(
            n_estimators=int(n_estimators),
            max_samples="auto",
            contamination=float(contamination),
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

        with st.spinner("Training Isolation Forest..."):
            X = df_train[num_cols + cat_cols]
            pipe.fit(X)
            # scores for training set
            scores = -pipe["model"].score_samples(pipe["prep"].transform(X))

        st.success("Model trained.")
        st.write(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

        # Training preview / metrics if labels exist
        if "is_fraud" in df_train.columns:
            y_true = df_train["is_fraud"].fillna(0).astype(int).values
            metrics = compute_metrics(y_true, scores)
            st.write("Metrics on training data (if labels exist):")
            st.json(metrics)

        # Save model and offer download
        model_dir = Path("models"); model_dir.mkdir(parents=True, exist_ok=True)
        dump(pipe, model_dir / "isoforest.pkl")
        st.success(f"Saved model to {model_dir / 'isoforest.pkl'}")

        # Download button
        bio = io.BytesIO()
        dump(pipe, bio)
        bio.seek(0)
        st.download_button("Download trained model (.pkl)", data=bio, file_name="isoforest.pkl")

        # Show score histogram
        import matplotlib.pyplot as plt
        fig2 = plt.figure()
        plt.hist(scores, bins=50)
        plt.title("Training-set anomaly score distribution")
        plt.xlabel("anomaly_score")
        plt.ylabel("count")
        st.pyplot(fig2)

# ------------------------------
# Footer
# ------------------------------
st.caption('Tip: Extend features with network/geo signals (e.g., shared phone/bank; rainfall around claim date) for stronger separation.')
