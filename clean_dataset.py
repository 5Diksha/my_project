# clean_dataset.py
import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    "train": DATA_DIR/"tweeteval_train.csv",
    "valid": DATA_DIR/"tweeteval_validation.csv",
    "test": DATA_DIR/"tweeteval_test.csv"
}

def normalize(text):
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\brt\b", " ", s, flags=re.I)
    s = re.sub(r"@\w+", " ", s)
    s = s.replace("#", "")
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

for split, path in paths.items():
    if not path.exists():
        print(f"Missing file: {path}. Make sure tweeteval CSVs are in data/")
        continue

    df = pd.read_csv(path)

    # find text column
    text_col = None
    for c in df.columns:
        if c.lower() in ("text", "tweet", "content"):
            text_col = c; break
    if text_col is None:
        text_col = df.columns[0]
    df = df.rename(columns={text_col: "text"})

    # find label column
    if "label" not in df.columns:
        for c in df.columns:
            if "label" in c.lower() or "sent" in c.lower() or "target" in c.lower():
                df = df.rename(columns={c: "label"}); break

    df["text_norm"] = df["text"].astype(str).apply(normalize)

    # drop rows with empty normalized text
    before = len(df)
    df = df[df["text_norm"].str.strip().astype(bool)].copy()
    dropped_empty = before - len(df)

    # dedupe (keep first occurrence)
    df["dupe_key"] = df["text_norm"].str.lower().str[:250]
    before = len(df)
    df = df.drop_duplicates(subset=["dupe_key"]).reset_index(drop=True)
    dropped_dupes = before - len(df)

    out_file = OUT_DIR / f"clean_{split}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved -> {out_file} (rows: {len(df)}). Dropped empty: {dropped_empty}, dropped dupes: {dropped_dupes}")
