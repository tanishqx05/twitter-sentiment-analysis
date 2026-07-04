# Twitter Sentiment Analysis System

A machine learning system that classifies tweets and social media text as **Positive**, **Neutral**, or **Negative** sentiment. Built with TF-IDF vectorization and a LinearSVC classifier, trained on 61,000+ real tweets.

## Live Demo

🔗 **[Try it here](https://twitter-sentiment-analysis-ep2oyeyp9seebrk3dt9pcd.streamlit.app/)**

## Results

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| **LinearSVC** | **98.5%** | **0.9855** |
| Logistic Regression | 98.1% | 0.9807 |
| Naive Bayes | 95.5% | 0.9553 |

## Tech Stack

Python · Scikit-Learn · TF-IDF · LinearSVC · Streamlit

## Dataset

[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) — 74,682 tweets across 4 sentiment classes. "Irrelevant" class dropped, leaving 61,121 tweets across Positive, Neutral, and Negative.

## How It Works

1. **Text Cleaning** — removes URLs, @mentions, special characters, lowercases
2. **TF-IDF Vectorization** — converts text to numerical features using unigrams + bigrams (50k features)
3. **LinearSVC Classification** — fast, accurate linear classifier with class balancing
4. **Confidence Scores** — decision function scores normalized for display

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

sentiment_project/
├── app/
│   ├── app.py                # Streamlit UI
│   ├── sentiment_model.pkl   # trained pipeline
│   ├── requirements.txt
│   └── model_comparison.csv
├── data/
│   ├── train_model.py        # training script
│   ├── twitter_training.csv
│   └── twitter_validation.csv
└── README.md

## Author

Tejasv Rathore — [GitHub](https://github.com/Tejasv1910)
