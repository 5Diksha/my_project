# my_project
X (twitter) Sentimental Analysis App

A machine learning project for classifying tweets into Positive, Neutral, and Negative sentiments using a hybrid architecture combining Sentence-BERT embeddings, VADER sentiment scores, and a LightGBM classifier.
This project includes dataset cleaning, model training, evaluation, visualizations, and a Streamlit web application for real-time predictions.

* Project Overview

This project performs sentiment analysis on Twitter (X) data using a state-of-the-art approach:

~ SBERT (Sentence-BERT)

Generates semantically rich embeddings for each tweet.

~ VADER (Valence Aware Dictionary)

Provides lexicon-based sentiment scoring, helping in short informal tweets and emojis.

~ LightGBM Classifier

Efficient gradient boosting algorithm used to classify sentiments based on SBERT + VADER features.

This hybrid method improves prediction stability and handles noisy tweet text effectively.

* Dataset

Dataset used: TweetEval – Sentiment Subset
Available splits:

Train: 45,562 samples

Validation: 2,000 samples

Test: 12,278 samples

Labels:

0 → Negative

1 → Neutral

2 → Positive

Cleaned datasets saved as:

clean_train.csv

clean_valid.csv

clean_test.csv

* Preprocessing Steps

Lowercasing & normalization

Removing links, special characters

Tokenization & Lemmatization

Stopword removal

Adding VADER sentiment score

Final merged feature matrix: 384-d SBERT + 1 VADER = 385 features

* Model Architecture
1 SBERT Embedding Extraction

Model used: all-MiniLM-L6-v2

Generates 384-dimensional embeddings

Captures semantic meaning of tweets

2 VADER Sentiment Score

Adds an additional numeric feature

Helps detect polarity from emojis, slang, and short text

3 LightGBM Classifier

Trained on the fused feature space

Fast and memory efficient

Good performance without GPU requirement

* Performance
Metric	Validation	Test
Accuracy	~67.65%	~65.66%

Key insights:

Neutral class is most challenging

Positive and Negative separation improves with SBERT

VADER strengthens polarity detection

* Visualizations Included

Class distribution plot

VADER score distribution

Confusion matrix

Precision–Recall–F1 bar chart

t-SNE/SBERT embedding scatter plot (2D)

All figures are stored in the /figs folder (or generated using make_plots.py).

* How to Run Locally
1. Create Virtual Environment
py -m venv venv
venv\Scripts\activate

2. Install Requirements
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run streamlit_app.py

* Project Demo

(If deployed on Streamlit Cloud, paste the link here)
COMING SOON

* Project Structure
my_project/
│
├── data/
│   ├── clean_train.csv
│   ├── clean_valid.csv
│   └── clean_test.csv
│
├── models/
│   ├── lgbm_model_quick.joblib
│   └── embedder_quick.joblib
│
├── streamlit_app.py
├── train_sbert_lgbm_quick.py
├── clean_dataset.py
├── make_plots.py
├── download_models.py
├── README.md
└── requirements.txt

* Download Model Weights

If model files are large, they can be downloaded automatically:

py download_models.py


(Add your Google Drive or GitHub Release links inside the script.)

* Future Enhancements

Fine-tuning full transformer models (BERT, RoBERTa)

Sarcasm and irony detection

Android app deployment

Live Twitter stream sentiment dashboard

Backend API using FastAPI

* Credits
Dataset: TweetEval Benchmark
SBERT: UKP Lab
VADER: Hutto & Gilbert
LightGBM: Microsoft Research
