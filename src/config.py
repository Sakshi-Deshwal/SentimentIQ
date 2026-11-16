
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Models/reports can be redirected (used by smoke tests so real artifacts are never touched).
ARTIFACT_ROOT = Path(os.environ.get("SENTIMENTIQ_ARTIFACT_DIR", ROOT))

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"          # women's clothing CSV lives here (Amazon is read from its local path)
MODELS_ROOT = ARTIFACT_ROOT / "models"
REPORTS_ROOT = ARTIFACT_ROOT / "reports"

# Folder names are kept as "three_class" so existing trained models and reports keep working.
THREE_MODELS_DIR = MODELS_ROOT / "three_class"
THREE_REPORTS_DIR = REPORTS_ROOT / "three_class"
ROBUSTNESS_PATH = ROOT / "tests_data" / "robustness_reviews.csv"
ROBUSTNESS_RESULTS_PATH = THREE_REPORTS_DIR / "robustness_results.csv"
ROBUSTNESS_SUMMARY_PATH = THREE_REPORTS_DIR / "robustness_summary.json"

# ---- columns ------------------------------------------------------------------------------
TEXT_COL = "review_text"
TARGET_COL = "sentiment"
ID_COL = "review_id"

# Columns that must NEVER be model inputs: only review_text is.
FORBIDDEN_FEATURES = {"rating", "review_id", "source", "domain", "sentiment", "split",
                      "category", "product_name"}

# ---- labels ---------------------------------------------------------------------------------
# Sorted alphabetically so they match scikit-learn's ``classes_`` ordering.
THREE_LABELS = ["Negative", "Neutral", "Positive"]
LABELS = THREE_LABELS
SPLITS = ["train", "validation", "test"]

SENTIMENT_COLORS = {"Positive": "#2E8B57", "Negative": "#C0392B", "Neutral": "#7F8C8D"}

# ---- models --------------------------------------------------------------------------------
MODEL_KEYS = ["naive_bayes", "logistic_regression", "linear_svm"]     # traditional ML
ML_KEYS = MODEL_KEYS
NN_KEYS = ["mlp", "textcnn", "bilstm"]
ALL_MODEL_KEYS = ML_KEYS + NN_KEYS
MODEL_NAMES = {
    "naive_bayes": "Multinomial Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "mlp": "MLP (TF-IDF)",
    "textcnn": "TextCNN",
    "bilstm": "BiLSTM",
}
MODEL_FAMILY = {k: "Traditional ML" for k in ML_KEYS} | {k: "Neural network" for k in NN_KEYS}

RANDOM_STATE = 42

# Streamlit upload limit for the Bulk page. Deliberately separate from local dataset loading,
# which has no size cap (the ~197 MB Amazon file is read from disk, never uploaded).
UPLOAD_MAX_MB = 50
