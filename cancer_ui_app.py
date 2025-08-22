# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------
# Simulated Cancer Dataset
# -----------------------
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "age": np.random.randint(20, 80, n),
    "smoking": np.random.randint(0, 2, n),        # 0 = No, 1 = Yes
    "alcohol": np.random.randint(0, 2, n),        # 0 = No, 1 = Yes
    "exercise": np.random.randint(0, 2, n),       # 0 = No, 1 = Yes
    "cancer": np.random.randint(0, 2, n)          # 0 = No, 1 = Yes
})

X = data.drop("cancer", axis=1)
y = data["cancer"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------
# Streamlit App
# -----------------------
st.title("🧬 Cancer Risk Prediction App")

st.write("Enter the details below to predict cancer risk:")

age = st.slider("Age", 20, 80, 30)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
exercise = st.selectbox("Exercise Regularly", ["No", "Yes"])

# Convert inputs to model format
smoking_val = 1 if smoking == "Yes" else 0
alcohol_val = 1 if alcohol == "Yes" else 0
exercise_val = 1 if exercise == "Yes" else 0

input_data = np.array([[age, smoking_val, alcohol_val, exercise_val]])

# Prediction
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

if st.button("Predict"):
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Cancer (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Cancer (Probability: {prob:.2f})")
