import streamlit as st
import pickle
import os
import numpy as np
import time

# Set page configuration
st.set_page_config(page_title="Disease Prediction Model", layout="wide", page_icon="Images/Icon.png")

def add_logo():
    logo = "Images/logo.png" 
    col1, col2 = st.columns([5, 1.5]) 
    with col1:
        st.title("Disease Prediction Model🩺")  
    with col2:
        st.image(logo, width=200)  

add_logo()

# Load the saved models and scalers
heart_model = pickle.load(open('Saved Models/heart_disease_model.sav', 'rb'))
diabetes_model = pickle.load(open('Saved Models/diabetes_model.sav', 'rb'))
parkinson_model = pickle.load(open('Saved Models/parkinsons_model.sav', 'rb'))

heart_scaler = pickle.load(open('Saved Models/scaler_heart.sav', 'rb'))
diabetes_scaler = pickle.load(open('Saved Models/scaler_diabetes.sav', 'rb'))
parkinson_scaler = pickle.load(open('Saved Models/scaler_parkinsons.sav', 'rb'))

# Function to predict heart disease
def predict_heart_disease(features):
    features_scaled = heart_scaler.transform([features])
    prediction = heart_model.predict(features_scaled)
    return prediction

# Function to predict diabetes
def predict_diabetes(features):
    features_scaled = diabetes_scaler.transform([features])
    prediction = diabetes_model.predict(features_scaled)
    return prediction

# Function to predict Parkinson's disease
def predict_parkinson(features):
    features_scaled = parkinson_scaler.transform([features])
    prediction = parkinson_model.predict(features_scaled)
    return prediction

# Add these helper functions after the initial imports
def show_feature_info(disease_type):
    info_mapping = {
        'heart': {
            'sex': "0 = Female, 1 = Male",
            'cp': "0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic",
            'fbs': "0 = False (< 120 mg/dl), 1 = True (> 120 mg/dl)",
            'restecg': "0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy",
            'exang': "0 = No, 1 = Yes",
            'slope': "0 = Upsloping, 1 = Flat, 2 = Downsloping",
            'thal': "1 = Normal, 2 = Fixed defect, 3 = Reversible defect"
        },
        'diabetes': {
            'pregnancies': "Number of times pregnant",
            'glucose': "Plasma glucose concentration (2 hours in an oral glucose tolerance test)",
            'blood_pressure': "Diastolic blood pressure (mm Hg)",
            'skin_thickness': "Triceps skin fold thickness (mm)",
            'insulin': "2-Hour serum insulin (mu U/ml)",
            'bmi': "Body mass index (weight in kg/(height in m)²)",
            'diabetes_pedigree': "Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)"
        },
        'parkinsons': {
            'MDVP': "Multiple Diagnostic Voice Program measurements",
            'Jitter/Shimmer': "Variations in fundamental frequency/amplitude",
            'NHR/HNR': "Noise to Harmonic Ratio/Harmonics to Noise Ratio",
            'RPDE': "Recurrence Period Density Entropy",
            'DFA': "Detrended Fluctuation Analysis",
            'spread1/spread2': "Nonlinear measures of fundamental frequency variation",
            'D2': "Correlation dimension",
            'PPE': "Pitch Period Entropy"
        }
    }
    return info_mapping.get(disease_type, {})

# App interface
tabs = st.tabs(["Home", "Heart Disease Prediction", "Diabetes Prediction", "Parkinson's Prediction"])

with tabs[0]:
    st.title("Welcome to the Disease Prediction Web App")
    st.markdown("""
    ### About the Web App
    This application uses Machine Learning models to predict the likelihood of:
    - **Heart Disease**
    - **Diabetes**
    - **Parkinson's Disease**
    
    ### How to Use the Web App
    1. Navigate to the respective tabs for Heart, Diabetes, or Parkinson's predictions
    2. Fill in the required input features in the form
    3. Click **Diagnose** to see the result
    4. Use the "?" icons next to each input to understand the features better

    ### Important Note
    ⚠️ This tool is for preliminary screening only and should not replace professional medical advice.
    Please consult with healthcare professionals for accurate diagnosis and treatment.

    ### Features
    - Easy-to-use interface
    - Instant predictions
    - Detailed feature explanations
    - Multiple disease prediction capabilities
    """)
    
    # Updated image rendering
    st.image("Images/Background.jpg", use_container_width=True, caption="YOUR HEALTH MATTERS")

# Heart Disease Prediction Tab
with tabs[1]:
    st.header("Heart Disease Prediction🫀")
    feature_info = show_feature_info('heart')
    
    with st.form(key='heart_form'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, step=1, help="Patient's age in years")
            sex = st.selectbox("Sex", [0, 1], help=feature_info['sex'])
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help=feature_info['cp'])
            trestbps = st.number_input("Resting Blood Pressure", min_value=0, step=1)
            chol = st.number_input("Serum Cholesterol", min_value=0, step=1)
            fbs = st.selectbox("Fasting Blood Sugar", [0, 1], help=feature_info['fbs'])
            restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], help=feature_info['restecg'])
        with col2:
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, step=1)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], help=feature_info['exang'])
            oldpeak = st.number_input("Depression Induced by Exercise", min_value=0.0, step=0.1)
            slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2], help=feature_info['slope'])
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [1, 2, 3], help=feature_info['thal'])
        
        diagnose_button = st.form_submit_button(label="Diagnose")
        if diagnose_button:
            with st.spinner('Analyzing... Please wait.'):
                time.sleep(2)
                features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                prediction = predict_heart_disease(features)
                confidence = heart_model.predict_proba([heart_scaler.transform([features])])[0]
                
                if prediction == 1:
                    st.error(f"Risk of Heart Disease Detected (Confidence: {confidence[1]:.2%})", icon="⚠️")
                    st.warning("""
                    **Recommended Next Steps:**
                    1. Consult a cardiologist
                    2. Get a complete cardiac evaluation
                    3. Review your lifestyle factors
                    """)
                else:
                    st.success(f"No Risk of Heart Disease Detected (Confidence: {confidence[0]:.2%})", icon="✅")
                    st.info("Continue maintaining a healthy lifestyle and regular check-ups")

# Diabetes Prediction Tab
with tabs[2]:
    st.header("Diabetes Prediction🩸")
    feature_info = show_feature_info('diabetes')
    
    with st.form(key='diabetes_form'):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, step=1, help=feature_info['pregnancies'])
            glucose = st.number_input("Glucose", min_value=0, step=1, help=feature_info['glucose'])
            blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1, help=feature_info['blood_pressure'])
            skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1, help=feature_info['skin_thickness'])
            insulin = st.number_input("Insulin", min_value=0, step=1, help=feature_info['insulin'])
        with col2:
            bmi = st.number_input("BMI", min_value=0.0, step=0.1, help=feature_info['bmi'])
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1, help=feature_info['diabetes_pedigree'])
            age = st.number_input("Age", min_value=0, max_value=100, step=1)

        diagnose_button = st.form_submit_button(label="Diagnose")
        if diagnose_button:
            with st.spinner('Analyzing... Please wait.'):
                time.sleep(2)
                features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
                prediction = predict_diabetes(features)
                confidence = diabetes_model.predict_proba([diabetes_scaler.transform([features])])[0]
                
                if prediction == 1:
                    st.error(f"Risk of Diabetes Detected (Confidence: {confidence[1]:.2%})", icon="⚠️")
                    st.warning("""
                    **Recommended Next Steps:**
                    1. Consult a diabetes specialist
                    2. Get a complete diabetes evaluation
                    3. Review your lifestyle factors
                    """)
                else:
                    st.success(f"No Risk of Diabetes Detected (Confidence: {confidence[0]:.2%})", icon="✅")
                    st.info("Continue maintaining a healthy lifestyle and regular check-ups")

# Parkinson's Disease Prediction Tab
with tabs[3]:
    st.header("Parkinson's Disease Prediction🧠")
    feature_info = show_feature_info('parkinsons')
    
    with st.form(key='parkinson_form'):
        col1, col2 = st.columns(2)
        with col1:
            MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1, help=feature_info['MDVP'])
            MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1, help=feature_info['MDVP'])
            MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1, help=feature_info['MDVP'])
            MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001, format="%.6f", help=feature_info['Jitter/Shimmer'])
            MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, step=0.001, format="%.6f", help=feature_info['Jitter/Shimmer'])
            MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.0, step=0.001, format="%.6f", help=feature_info['MDVP'])
            MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.0, step=0.001, format="%.6f", help=feature_info['MDVP'])
            Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.0, step=0.001, format="%.6f", help=feature_info['Jitter/Shimmer'])
            MDVP_Shim = st.number_input("MDVP:Shimmer", min_value=0.0, step=0.001, format="%.6f", help=feature_info['Jitter/Shimmer'])
            MDVP_Shim_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, step=0.1, help=feature_info['Jitter/Shimmer'])
            Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, step=0.001, format="%.6f", help=feature_info['Jitter/Shimmer'])
        with col2:
            Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, step=0.001, format="%.6f", help=feature_info['Jitter/Shimmer'])
            MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0, step=0.001, format="%.6f", help=feature_info['MDVP'])
            Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.0, step=0.001, format="%.6f", help=feature_info['Jitter/Shimmer'])
            NHR = st.number_input("NHR", min_value=0.0, step=0.001, format="%.6f", help=feature_info['NHR/HNR'])
            HNR = st.number_input("HNR", min_value=0.0, step=0.1, help=feature_info['NHR/HNR'])
            RPDE = st.number_input("RPDE", min_value=0.0, max_value=1.0, step=0.001, format="%.6f", help=feature_info['RPDE'])
            DFA = st.number_input("DFA", min_value=0.0, max_value=1.0, step=0.001, format="%.6f", help=feature_info['DFA'])
            spread1 = st.number_input("Spread1", min_value=-10.0, max_value=1.0, step=0.001, format="%.6f", help=feature_info['spread1/spread2'])
            spread2 = st.number_input("Spread2", min_value=-1.0, max_value=1.0, step=0.001, format="%.6f", help=feature_info['spread1/spread2'])
            D2 = st.number_input("D2", min_value=0.0, step=0.001, format="%.6f", help=feature_info['D2'])
            PPE = st.number_input("PPE", min_value=0.0, step=0.001, format="%.6f", help=feature_info['PPE'])

        diagnose_button = st.form_submit_button(label="Diagnose")
        if diagnose_button:
            with st.spinner('Analyzing... Please wait.'):
                time.sleep(2)
                features = [
                    MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ,
                    Jitter_DDP, MDVP_Shim, MDVP_Shim_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
                    NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
                ]
                prediction = predict_parkinson(features)
                confidence = parkinson_model.predict_proba([parkinson_scaler.transform([features])])[0]
                
                if prediction == 1:
                    st.error(f"Risk of Parkinson's Disease Detected (Confidence: {confidence[1]:.2%})", icon="⚠️")
                    st.warning("""
                    **Recommended Next Steps:**
                    1. Consult a neurologist
                    2. Get a complete neurological evaluation
                    3. Review your lifestyle factors
                    """)
                else:
                    st.success(f"No Risk of Parkinson's Disease Detected (Confidence: {confidence[0]:.2%})", icon="✅")
                    st.info("Continue maintaining a healthy lifestyle and regular check-ups")
