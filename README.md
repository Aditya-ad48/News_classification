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
