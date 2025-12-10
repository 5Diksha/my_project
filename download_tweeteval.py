# download_tweeteval.py
from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
ds['train'].to_csv("data/tweeteval_train.csv", index=False)
ds['validation'].to_csv("data/tweeteval_validation.csv", index=False)
ds['test'].to_csv("data/tweeteval_test.csv", index=False)

print("Saved CSVs to ./data/: tweeteval_train.csv, tweeteval_validation.csv, tweeteval_test.csv")
