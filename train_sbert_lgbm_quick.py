# train_sbert_lgbm_quick.py
# Fast SBERT + LightGBM with VADER and class weights, tuned for speed.
import os
import joblib
import math
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
tqdm.pandas()

# VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

DATA_DIR = Path("data")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

# Load cleaned data
train = pd.read_csv(DATA_DIR/"clean_train.csv")
valid = pd.read_csv(DATA_DIR/"clean_valid.csv")
test  = pd.read_csv(DATA_DIR/"clean_test.csv")

# Ensure text_norm exists
for df in (train, valid, test):
    if "text_norm" not in df.columns:
        df["text_norm"] = df.get("text", df.iloc[:,0]).astype(str)

# Robust label mapping
def map_labels(col):
    vals = sorted(col.dropna().unique())
    if all(isinstance(v, (int, np.integer)) for v in vals):
        return None
    mapping = {}
    for v in vals:
        s = str(v).lower()
        if "pos" in s:
            mapping[v] = 2
        elif "neg" in s:
            mapping[v] = 0
        elif "neu" in s:
            mapping[v] = 1
        else:
            try:
                mapping[v] = int(v)
            except:
                mapping[v] = 1
    return mapping

label_map = map_labels(train["label"])
if label_map:
    train["y"] = train["label"].map(label_map).astype(int)
    valid["y"] = valid["label"].map(label_map).astype(int)
    test["y"]  = test["label"].map(label_map).astype(int)
else:
    train["y"] = train["label"].astype(int)
    valid["y"] = valid["label"].astype(int)
    test["y"]  = test["label"].astype(int)

print("Sizes (train/valid/test):", len(train), len(valid), len(test))
print("Train label distribution:\n", train["y"].value_counts().to_string())

# Compute VADER compound score
def vader_compound(s):
    try:
        return float(sia.polarity_scores(str(s))["compound"])
    except:
        return 0.0

print("Computing VADER scores...")
train["vader"] = train["text_norm"].progress_apply(vader_compound)
valid["vader"] = valid["text_norm"].progress_apply(vader_compound)
test["vader"]  = test["text_norm"].progress_apply(vader_compound)

# Load SBERT embedder
print("Loading SBERT (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode (may be the slowest step)
batch_size = 32
print("Encoding train...")
X_train_emb = embedder.encode(train["text_norm"].tolist(), batch_size=batch_size, show_progress_bar=True)
print("Encoding valid...")
X_valid_emb = embedder.encode(valid["text_norm"].tolist(), batch_size=batch_size, show_progress_bar=True)
print("Encoding test...")
X_test_emb  = embedder.encode(test["text_norm"].tolist(), batch_size=batch_size, show_progress_bar=True)

# Attach VADER feature
X_train = np.hstack([X_train_emb, train["vader"].values.reshape(-1,1)])
X_valid = np.hstack([X_valid_emb, valid["vader"].values.reshape(-1,1)])
X_test  = np.hstack([X_test_emb,  test["vader"].values.reshape(-1,1)])

y_train = train["y"].values
y_valid = valid["y"].values
y_test  = test["y"].values

# Compute class weights (inverse sqrt freq)
freq = np.bincount(y_train)
class_weight = {i: 1.0/math.sqrt(int(freq[i])) for i in range(len(freq))}
weights = np.array([class_weight[int(y)] for y in y_train])

print("Class freq:", dict(enumerate(freq)))
print("Class weights:", class_weight)

# LightGBM params (faster config)
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "learning_rate": 0.04,
    "num_leaves": 48,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42
}

dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

# Train with callback-based early stopping (compatible with all versions)
print("Training LightGBM (max 300 rounds, early stopping 40)...")
callbacks = [
    lgb.early_stopping(stopping_rounds=40, verbose=True),
    lgb.log_evaluation(period=40)
]

bst = lgb.train(
    params,
    dtrain,
    num_boost_round=300,
    valid_sets=[dvalid],
    callbacks=callbacks
)

# Evaluate
probs_valid = bst.predict(X_valid)
pred_valid = np.argmax(probs_valid, axis=1)

probs_test = bst.predict(X_test)
pred_test = np.argmax(probs_test, axis=1)

print("\nVALIDATION ACCURACY:", round(accuracy_score(y_valid, pred_valid)*100,2), "%")
print(classification_report(y_valid, pred_valid, digits=4))

print("\nTEST ACCURACY:", round(accuracy_score(y_test, pred_test)*100,2), "%")
print(classification_report(y_test, pred_test, digits=4))

# Save artifacts
OUT = OUT_DIR
joblib.dump(embedder, OUT/"embedder_quick.joblib")
joblib.dump(bst, OUT/"lgbm_model_quick.joblib")
print("\nSaved models to:", OUT)
