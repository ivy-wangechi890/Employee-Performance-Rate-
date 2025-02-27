import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('rf.pkl')

st.cache_resource.clear()

# Custom CSS
st.markdown("""
    <style>
        body { background-color: #f4f7f9; font-family: 'Arial', sans-serif; }
        .stApp { color: #0d1b2a; }
        .stButton>button { background-color: #0a3d62; color: white; font-size: 16px; padding: 12px; border-radius: 8px; border: none; }
        .stButton>button:hover { background-color: #1c5980; }
        .stTextInput>div>input, .stNumberInput input, .stSelectbox select { 
            border: 1px solid #0a3d62; border-radius: 5px; padding: 8px; font-size: 14px;
        }
        h1, h2, h3, h4 { color: #0a3d62; }
        .result-box { background-color: #e8f0f7; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("Employee Performance Rating Tool")
st.write("""
A data-driven approach to analyze key factors influencing employee performance.  
Evaluate work-life balance, employee environment, salary hikes, experience, and managerial relationships to 
gain insights that improve **engagement, productivity, and retention.**
""")
st.markdown("---")

# Load the model
model = load_model()

if model:
    st.header("Enter Employee Details")
    
    # Input Form Layout
    col1, col2 = st.columns(2)
    
    with col1:
        EmpDepartment = st.selectbox("Employee Department", options=[
            (0, "Sales"), (1, "Human Resources"), (2, "Development"), 
            (3, "Data Science"), (4, "Research and Development"), (5, "Finance")
        ], format_func=lambda x: x[1])[0]

        EmpLastSalaryHikePercent = st.number_input("Last Salary Hike (%)", min_value=0, max_value=100, value=0)

        EmpWorkLifeBalance = st.selectbox("Work-Life Balance", options=[
            (0, "Poor"), (1, "Average"), (2, "Excellent")
        ], format_func=lambda x: x[1])[0]
        
        EmpEnvironmentSatisfaction = st.selectbox("Environment Satisfaction", options=[
            (0, "Poor"), (1, "Average"), (2, "Excellent")
        ], format_func=lambda x: x[1])[0]

        EmpJobRole_encoded = st.selectbox("Employee Job Role", options=[
            (0, "Sales Executive"), (1, "Manager"), (2, "Developer"), 
            (3, "Sales Representative"), (4, "Human Resources"), (5, "Senior Developer"),
            (6, "Data Scientist"), (7, "Senior Manager R&D"), (8, "Laboratory Technician"), 
            (9, "Manufacturing Director"), (10, "Research Scientist"), (11, "Healthcare Representative"), 
            (12, "Research Director"), (13, "Manager R&D"), (14, "Finance Manager"), 
            (15, "Technical Architect"), (16, "Business Analyst"), (17, "Technical Lead"), 
            (18, "Delivery Manager")
        ], format_func=lambda x: x[1])[0]

    with col2:
        ExperienceYearsAtThisCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=0)
        ExperienceYearsInCurrentRole = st.number_input("Years in Current Role", min_value=0, max_value=40, value=0)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=40, value=0)
        YearsWithCurrManager = st.number_input("Years with Current Manager", min_value=0, max_value=40, value=0)

    
      

    st.markdown("---")

    # Prepare input data
    input_data = np.array([
    EmpDepartment, EmpLastSalaryHikePercent, EmpWorkLifeBalance, 
    EmpEnvironmentSatisfaction, EmpJobRole_encoded,  # Add this!
    ExperienceYearsAtThisCompany, ExperienceYearsInCurrentRole, 
    YearsSinceLastPromotion, YearsWithCurrManager
])


   

    input_data = input_data.reshape(1, -1)  # Reshape for model

    # Debugging: Check feature count
    expected_features = model.n_features_in_
    if input_data.shape[1] != expected_features:
        st.error(f"Feature mismatch: Model expects {expected_features} features, but received {input_data.shape[1]}.")
    else:
        # Prediction Button
        if st.button("Predict Performance Rating", key="predict_button"):
            prediction = model.predict(input_data)
            performance_rating = "Poor" if prediction[0] == 0 else "Average" if prediction[0] == 1 else "Excellent"
            
            # Display Prediction
            st.markdown(f'<div class="result-box">Predicted Performance Rating: <strong>{performance_rating}</strong></div>', unsafe_allow_html=True)

            # Actionable Insights
            st.markdown("#### Insights & Recommendations")
            st.write("""
            - **Enhance Employee Satisfaction**: Address work-life balance and environment satisfaction.  
            - **Fair Compensation**: Competitive salary hikes encourage retention.  
            - **Career Development**: Provide career growth opportunities and regular promotions.  
            - **Managerial Support**: Encourage stronger manager-employee relationships.  
            """)

st.markdown("---")
st.write("**Actionable Recommendations:** Focus on employee work environment, fair salary hikes, and career growth to enhance performance.")

