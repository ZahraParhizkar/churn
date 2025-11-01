import streamlit as st
import pandas as pd
import joblib

# مدل را لود کن
model = joblib.load('finalizedchurn.sav')

# 📁 لیست تمام فیچرهایی که مدل با آن آموزش دیده (در زمان training ذخیره کن)
feature_names = model.feature_names_in_

st.title("💡 پیش‌بینی ترک مشتری (Customer Churn Prediction)")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ['Female', 'Male'])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ['Yes', 'No'])
    Dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure (ماه)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    MultipleLines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    TechSupport = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2500.0)
    submitted = st.form_submit_button("🔮 پیش‌بینی کن")

if submitted:
    # ساخت ورودی خام
    input_dict = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }

    df_input = pd.DataFrame(input_dict)

    # 🧩 اجرای one-hot encoding دقیقاً مثل زمان آموزش
    df_input_encoded = pd.get_dummies(df_input)

    # اضافه کردن ستون‌های گمشده
    for col in feature_names:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    # تنظیم ترتیب ستون‌ها مطابق مدل
    df_input_encoded = df_input_encoded[feature_names]

    # 🔮 پیش‌بینی
    prediction = model.predict(df_input_encoded)

    st.subheader("نتیجه پیش‌بینی:")
    if prediction[0] == 1 or prediction[0] == 'Yes':
        st.error("🚨 این مشتری احتمالاً **ترک خواهد کرد** (Churn = Yes)")
    else:
        st.success("✅ این مشتری احتمالاً **باقی خواهد ماند** (Churn = No)")
