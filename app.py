import streamlit as st
import joblib
import re

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text


st.title("Fake News Detection System")

user_input = st.text_area("Enter News Text")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter text!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        prob = model.predict_proba(vectorized)

        if prediction[0] == "FAKE":
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        st.write(f"Fake Probability: {prob[0][0]:.2f}")
        st.write(f"Real Probability: {prob[0][1]:.2f}")