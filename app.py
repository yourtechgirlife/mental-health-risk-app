
# your Streamlit code goes here
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample app
st.title("Mental Health Risk Prediction for First-Year Students")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview", data.head())

    # Continue with preprocessing, model training, etc.
    if st.button("Train Model"):
        # Assuming 'label' is the target
        X = data.drop("label", axis=1)
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model Accuracy: {acc:.2f}")
