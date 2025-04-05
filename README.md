# Fake News Classification Project

## Overview
This project implements a machine learning pipeline to classify news articles as **real (1)** or **fake (0)** based on their text content. It uses natural language processing (NLP) techniques and evaluates multiple classification algorithms, culminating in an ensemble `VotingClassifier` that combines Support Vector Classifier (SVC), Multinomial Naive Bayes (MNB), and Extra Trees Classifier (ETC). The project achieves high accuracy and precision, making it a robust tool for fake news detection.

Developed in Python using a Jupyter Notebook, it leverages libraries like `pandas`, `numpy`, `nltk`, `seaborn`, `matplotlib`, and `scikit-learn`.

## 📥 Download Dataset

Due to GitHub file size limitations, the dataset (`train.tsv`) is hosted externally:

🔗 [Download train.tsv from Google Drive](https://drive.google.com/uc?id=YOUR_FILE_ID)

Once downloaded, place it in the root directory of this project.

## Dataset
 
🔗 [Download train.tsv from Google Drive](https://drive.google.com/file/d/1MVCwrTyZigkhJi-bksa_GxpmvcqFUB5I/view?usp=drive_link)

- **Format**: Tab-separated values (TSV)
- **Columns**:
  - `Unnamed: 0`: Index (dropped)
  - `title`: Article title (dropped)
  - `text`: Article content (used for classification)
  - `subject`: News category (dropped)
  - `date`: Publication date (dropped)
  - `label`: Target (0 = fake, 1 = real)
  - **Size**: 30,000 rows

## 🧠 Features

- Text preprocessing using **NLTK**: tokenization, cleaning, stopword removal
- Exploratory data analysis (EDA) using **matplotlib** and **seaborn**
- Feature vectorization using **TF-IDF**
- Classification using **Naive Bayes** and **Logistic Regression**
- Evaluation using metrics like **accuracy**, **confusion matrix**, **F1-score**

## 🗂️ Project Structure

```
├── news_classification.ipynb     
├── requirements.txt
├── voting_classifier_model.pkl            
├── README.md
                                         
```
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
