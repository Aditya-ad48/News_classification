import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string


import os


nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)


@st.cache_resource
def load_model():
    with open('voting_classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)


st.title("📰 News Category Classifier")
st.caption("🔥 A simple ML-powered app to classify news text into categories.")

user_input = st.text_area("Enter your news text here:")

if st.button("Classify"):
    if user_input:
        cleaned_text = preprocess_text(user_input)
        try:
            X_input = vectorizer.transform([cleaned_text])
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Category: **{prediction}**")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text to classify.")
