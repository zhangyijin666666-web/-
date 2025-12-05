import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set the banner
st.image("images.jpg", use_container_width=True)

# App title
st.title("Cardiovascular Disease Prediction App")
st.write("""
This application predicts the likelihood of cardiovascular disease based on patient data.
""")

# Load pre-trained models
model_files = {
    "SVM": "svm.pkl",
    "Random Forest": "random_forest.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Nueral Network": "neural_network.pkl"
}

loaded_models = {}
for name, file in model_files.items():
    with open(file, 'rb') as f:
        loaded_models[name] = pickle.load(f)

# Sidebar: Select the model
st.sidebar.header("Select Model and Enter Patient Data")
selected_model_name = st.sidebar.selectbox("Choose a trained model:", list(loaded_models.keys()))
selected_model = loaded_models[selected_model_name]

# Load the scaler
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)


def get_user_input():
    # 1) now collect gender
    gender = st.sidebar.selectbox(
        "Gender",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    weight = st.sidebar.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
    ap_hi = st.sidebar.number_input("Systolic Blood Pressure (ap_hi)", min_value=50, max_value=250, value=120)
    ap_lo = st.sidebar.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=150, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=35.0, value=25.0)
    cholesterol = st.sidebar.selectbox(
        "Cholesterol Level",
        [1, 2, 3],
        format_func=lambda x: ["Normal","Above Normal","Well Above Normal"][x-1]
    )
    gluc = st.sidebar.selectbox(
        "Glucose Level",
        [1, 2, 3],
        format_func=lambda x: ["Normal","Above Normal","Well Above Normal"][x-1]
    )
    smoke = st.sidebar.selectbox("Smokes?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    alco  = st.sidebar.selectbox("Drinks Alcohol?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    active= st.sidebar.selectbox("Physically Active?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    bp_category = st.sidebar.selectbox(
        "BP Category",
        ["Normal","Elevated","Hypertension Stage 1","Hypertension Stage 2"]
    )

    bp_mapping = {
        "Normal":                [0,0,0,1],
        "Elevated":              [0,1,0,0],
        "Hypertension Stage 1":  [1,0,0,0],
        "Hypertension Stage 2":  [0,0,1,0],
    }
    bp_encoded = bp_mapping[bp_category]

    data = {
        "age": age,
        "gender": gender,                      # ← include gender here
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "bmi": bmi,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bp_category_Hypertension Stage 1": bp_encoded[0],
        "bp_category_Elevated":              bp_encoded[1],
        "bp_category_Hypertension Stage 2": bp_encoded[2],
        "bp_category_Normal":                bp_encoded[3],
    }

    # 2) build DataFrame
    df = pd.DataFrame([data])

    # 3) (optional but fool‑proof) reorder to match exactly your training features
    feature_order = [
        "age","gender","weight","ap_hi","ap_lo","bmi",
        "cholesterol","gluc","smoke","alco","active",
        "bp_category_Hypertension Stage 1",
        "bp_category_Elevated",
        "bp_category_Hypertension Stage 2",
        "bp_category_Normal",
    ]
    df = df[feature_order]

    return df


# Collect user input
user_data = get_user_input()

# Display user input
st.write("**Patient Data:**")
st.write(user_data)

# Prediction button
if st.button("Predict for Patient"):
    # Scale the user data
    scaled_data = scaler.transform(user_data)
    
    # Make prediction
    patient_prediction = selected_model.predict(scaled_data)[0]
    result = "Disease Detected" if patient_prediction == 1 else "No Disease"
    st.success(f"Prediction: {result}")

# Optional: Example Visualization
if st.checkbox("Show Example Visualization"):
    st.write("**Example Confusion Matrix for Selected Model:**")
    # Assume we have a confusion matrix from the selected model's test data
    y_true = np.random.choice([0, 1], size=100)  # Simulated true labels
    y_pred = np.random.choice([0, 1], size=100)  # Simulated predicted labels
    conf_matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
