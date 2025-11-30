import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🎯", layout="wide")

# Title
st.title("🎯 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn using machine learning")

# Sidebar
st.sidebar.header("📊 Input Customer Data")

# Load model (you'll need to train and save it first)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model_XGBoost.pkl')
        return model
    except:
        st.error("⚠️ Model not found! Please train the model first by running main.py")
        return None

model = load_model()

# Input fields (customize based on your dataset)
st.sidebar.subheader("Customer Information")

# Example inputs - adjust based on your actual features
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Profile")
    profile_data = {
        "Feature": ["Tenure", "Monthly Charges", "Total Charges", "Contract", 
                   "Payment Method", "Internet Service", "Online Security", "Tech Support"],
        "Value": [f"{tenure} months", f"${monthly_charges}", f"${total_charges}", 
                 contract, payment_method, internet_service, online_security, tech_support]
    }
    st.table(pd.DataFrame(profile_data))

with col2:
    st.subheader("🔮 Prediction")
    
    if model is not None:
        if st.button("Predict Churn", type="primary"):
            # Create feature vector (adjust based on your preprocessing)
            # This is a simplified example - you'll need to match your actual preprocessing
            
            # Encode categorical variables (simplified)
            contract_encoded = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
            payment_encoded = {"Electronic check": 0, "Mailed check": 1, 
                              "Bank transfer": 2, "Credit card": 3}[payment_method]
            
            # Create feature array (adjust to match your model's expected features)
            features = np.array([[tenure, monthly_charges, total_charges, 
                                contract_encoded, payment_encoded]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            # Display results
            if prediction == 1:
                st.error("⚠️ HIGH RISK: Customer likely to churn")
                st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
            else:
                st.success("✅ LOW RISK: Customer likely to stay")
                st.metric("Retention Probability", f"{probability[0]*100:.1f}%")
            
            # Probability bar
            st.progress(probability[1])
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.markdown("""
                - Offer retention discount or upgrade
                - Reach out with personalized support
                - Consider contract extension incentives
                - Improve service quality
                """)
            else:
                st.markdown("""
                - Continue excellent service
                - Consider upselling opportunities
                - Maintain regular engagement
                """)
    else:
        st.warning("Please train the model first by running: `python main.py`")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub](https://github.com/Akrati36/customer-churn-prediction)")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Train the model**: Run `python main.py` to train and save the model
    2. **Input customer data**: Use the sidebar to enter customer information
    3. **Get prediction**: Click 'Predict Churn' to see the results
    4. **Review recommendations**: Check suggested actions based on prediction
    
    **Note**: Adjust the input fields to match your dataset's features.
    """)