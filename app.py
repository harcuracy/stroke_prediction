import streamlit as st
import pandas as pd
import joblib

# --- 1. Load model ---
model_path = 'model/stroke_model_threshold_0_3.pkl'
model_data = joblib.load(model_path)
pipeline = model_data['pipeline']
threshold = model_data['threshold']

st.title("🧠 Stroke Prediction App")
st.write("Provide patient information to predict stroke risk.")

# --- 2. Extract preprocessor ---
preprocessor = pipeline.named_steps['preprocessor']

# Extract all expected features
all_features = []
for _, _, feature_list in preprocessor.transformers_:
    all_features.extend(feature_list)

# Extract categorical features and encoder categories
categorical_features = preprocessor.transformers_[1][2]
encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']

# Define which features are binary
binary_features = ["hypertension", "heart_disease"]

# Define which numeric features should be integers (like age)
integer_features = ["age"]

st.sidebar.header("Model Information")
st.sidebar.write("**Expected features:**")
st.sidebar.write(all_features)

# --- 3. Unified feature input section ---
st.subheader("Patient Information")
inputs = {}

for feature in all_features:

    # ---- Case 1: Binary features as Yes/No dropdown ----
    if feature in binary_features:
        inputs[feature] = st.selectbox(feature, ["No", "Yes"])

    # ---- Case 2: Other categorical features ----
    elif feature in categorical_features:
        idx = categorical_features.index(feature)

        if idx < len(encoder.categories_):
            cats = list(encoder.categories_[idx])
        else:
            cats = ["Unknown"]

        inputs[feature] = st.selectbox(feature, cats)

    # ---- Case 3: Integer numeric fields ----
    elif feature in integer_features:
        inputs[feature] = st.number_input(feature, value=0, step=1)

    # ---- Case 4: Other numeric fields ----
    else:
        inputs[feature] = st.number_input(feature, value=0.0)

# --- 4. Convert inputs to DataFrame ---
input_df = pd.DataFrame([inputs])

# --- 5. Convert Yes/No → 1/0 for binary fields ---
for bf in binary_features:
    if bf in input_df.columns:
        input_df[bf] = input_df[bf].map({"No": 0, "Yes": 1})

# --- 6. Predict ---
if st.button("Predict Stroke Risk"):
    try:
        probs = pipeline.predict_proba(input_df)[:, 1]
        y_pred = (probs >= threshold).astype(int)

        st.success(f"✅ Stroke Probability: **{probs[0]*100:.2f}%**")
        st.write(f"Prediction: **{'Stroke' if y_pred[0] == 1 else 'No Stroke'}**")
        st.progress(float(probs[0]))

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- 7. Footer ---
st.markdown("---")
st.caption(f"Model loaded from: `{model_path}`")
