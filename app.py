import streamlit as st
import pandas as pd
import numpy as np
import joblib
import inspect

# --- 1. Load model ---
model_path = 'model/stroke_model_threshold_0_3.pkl'  # adjust path as needed
model_data = joblib.load(model_path)
pipeline = model_data['pipeline']
threshold = model_data['threshold']

st.title("🧠 Stroke Prediction App")
st.write("Provide patient information to predict stroke risk.")

# --- 2. Automatically detect feature types ---
preprocessor = pipeline.named_steps['preprocessor']
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

st.sidebar.header("Model Information")
st.sidebar.write(f"**Expected numeric features:** {list(numeric_features)}")
st.sidebar.write(f"**Expected categorical features:** {list(categorical_features)}")
st.sidebar.write("Final year project")

# --- 3. Create UI dynamically based on expected columns ---
inputs = {}

# Numeric fields
st.subheader("Numeric Features")
for col in numeric_features:
    inputs[col] = st.number_input(f"{col}", value=0.0, step=0.1)

# Categorical fields
st.subheader("Categorical Features")
for col in categorical_features:
    # Attempt to infer categories from the trained OneHotEncoder
    encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    # Find index of this column in the categorical_features list
    idx = categorical_features.index(col)
    if idx < len(encoder.categories_):
        options = list(encoder.categories_[idx])
    else:
        options = ["Unknown"]
    inputs[col] = st.selectbox(f"{col}", options)

# --- 4. Convert inputs to DataFrame ---
input_df = pd.DataFrame([inputs])

# --- 5. Make prediction ---
if st.button("Predict Stroke Risk"):
    try:
        probs = pipeline.predict_proba(input_df)[:, 1]
        y_pred = (probs >= threshold).astype(int)

        st.success(f"✅ Predicted Probability of Stroke: {probs[0]*100:.2f}%")
        st.write(f"Predicted Class: **{'Stroke' if y_pred[0]==1 else 'No Stroke'}**")
        st.progress(float(probs[0]))
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- 6. Footer ---
st.markdown("---")
st.caption("Model loaded from: `" + model_path + "`")
