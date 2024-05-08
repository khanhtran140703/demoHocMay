import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("Stroke.mdl")

def predict_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # Prepare inputs for prediction
    new_patient = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
    
    # Perform prediction
    prediction = model.predict(new_patient)
    
    return prediction

def main():
    st.title("Stroke Prediction ML App")
    
    # Input fields
    gender = st.selectbox('Gender', ["Male", "Female"])
    age = st.slider('Age', 0, 100, 50)
    hypertension = st.selectbox('Hypertension', ["No", "Yes"])
    heart_disease = st.selectbox('Heart Disease', ["No", "Yes"])
    ever_married = st.selectbox('Ever Married', ["No", "Yes"])
    work_type = st.selectbox('Work Type', ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox('Residence Type', ["Urban", "Rural"])
    avg_glucose_level = st.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.slider('BMI', 0.0, 100.0, 25.0)
    smoking_status = st.selectbox('Smoking Status', ["Unknown", "Never smoked", "formerly smoked", "smokes"])
    
    # Convert categorical inputs to binary
    gender = 0 if gender == "Male" else 1
    ever_married = 0 if ever_married == "No" else 1
    hypertension = 0 if hypertension == "No" else 1
    heart_disease = 0 if heart_disease == "No" else 1
    work_type = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"].index(work_type)
    Residence_type = 0 if Residence_type == "Urban" else 1
    smoking_status = ["Unknown", "Never smoked", "formerly smoked", "smokes"].index(smoking_status)
    
    # Prediction button
    if st.button("Predict"):
        result = predict_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
        if result == 1:
            st.success("The patient is likely to have a stroke!")
        else:
            st.success("The patient is not likely to have a stroke!")

if __name__ == '__main__':
    main()
