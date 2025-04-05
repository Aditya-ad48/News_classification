# Fake News Classification Project

## Overview
This project implements a machine learning pipeline to classify news articles as **real (1)** or **fake (0)** based on their text content. It uses natural language processing (NLP) techniques and evaluates multiple classification algorithms, culminating in an ensemble `VotingClassifier` that combines Support Vector Classifier (SVC), Multinomial Naive Bayes (MNB), and Extra Trees Classifier (ETC). The project achieves high accuracy and precision, making it a robust tool for fake news detection.

Developed in Python using a Jupyter Notebook, it leverages libraries like `pandas`, `numpy`, `nltk`, `seaborn`, `matplotlib`, and `scikit-learn`.

## Dataset
- **File**: `train.tsv` (not included in the repository; provide your own)
- **Format**: Tab-separated values (TSV)
- **Columns**:
  - `Unnamed: 0`: Index (dropped)
  - `title`: Article title (dropped)
  - `text`: Article content (used for classification)
  - `subject`: News category (dropped)
  - `date`: Publication date (dropped)
  - `label`: Target (0 = fake, 1 = real)
  - **Size**: 30,000 rows
  
## 🎯 Demo

Here's a sneak peek at the pipeline:

| Step               | Description                                  |
|--------------------|----------------------------------------------|
| Data Preprocessing | Tokenization, stopword removal, cleaning     |
| Feature Extraction | TF-IDF, Count Vectorization                  |
| Model Training     | Multinomial Naive Bayes, Logistic Regression |
| Evaluation         | Confusion Matrix, Accuracy, Classification Report |

## 🗂️ Project Structure

news_classification/
│
├── Fake_News_Classification.ipynb  # Main notebook
├── voting_classifier_model.pkl     # Saved model (generated after running)
├── README.md                       # This file
└── requirements.txt                # Dependencies (optional)

## Model Details
- **Features**: Text data (preprocessed into TF-IDF vectors).
- **Target**: Binary classification (0 = fake, 1 = real).
- **Ensemble Model**:
  - SVC (sigmoid kernel, `gamma=1.0`, `probability=True`)
  - Multinomial Naive Bayes
  - Extra Trees Classifier (50 estimators, `random_state=2`)
  - **Voting**: Soft voting
- **Performance**:
  - Accuracy: ~0.968
  - Precision: ~0.965

## Results
The `VotingClassifier` outperforms individual models, achieving:
- **Accuracy**: 0.968
- **Precision**: 0.965

See the notebook for a detailed comparison of all tested algorithms.
