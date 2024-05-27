import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title='Predict Customer Churn - Telco',
    layout='wide'
)

def enter():
    st.markdown("<br>", unsafe_allow_html=True)
def horizontal_line():
    st.markdown("<hr>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; font-size:24px'>
            <b>
            Customer <br> Churn Prediction<br>
            </b>
        </div>
    """, unsafe_allow_html=True)
    
    horizontal_line()
    
    selected_option_menu = option_menu(menu_title=None, 
                        options=["Predict Churn", 'Dataset Preview'], 
                        icons=['house'], 
                        menu_icon="cast", default_index=0
                    )
    
    horizontal_line()
    
    st.markdown("""
        <div style='text-align: center; font-size:20px'>
            <b>Dataset Source</b> <br>
            <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" style="text-decoration: none;">Telco Customer Churn</a>
        </div>
    """, unsafe_allow_html=True)
    
    horizontal_line()

    st.markdown("""
        <div style='text-align: center; font-size:20px'>
            <b>Github Repository</b> <br>
            <a href="https://github.com/TheOX7/KPK-2" style="text-decoration: none;">Repository</a>
        </div>
    """, unsafe_allow_html=True)
    
    horizontal_line()
    
    st.markdown("""
        <div style='text-align: center; font-size:20px'>
            <b>Created By</b> <br>
            <a href="linkedin.com/in/marselius-agus-dhion/" style="text-decoration: none;">Marselius Agus Dhion</a>
        </div>
    """, unsafe_allow_html=True)
    
    horizontal_line()

enter()

if selected_option_menu == "Predict Churn":
    # Load model yang telah disimpan (didump)
    model = joblib.load('model.joblib')

    # Daftar kolom fitur sesuai dengan model yang telah dilatih
    feature_columns = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
        'gender_Male', 'SeniorCitizen_No', 'SeniorCitizen_Yes', 'Partner_No',
        'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
        'PhoneService_Yes', 'MultipleLines_No',
        'MultipleLines_No_phone_service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber_optic',
        'InternetService_No', 'OnlineSecurity_No',
        'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_No_internet_service',
        'OnlineBackup_Yes', 'DeviceProtection_No',
        'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
        'TechSupport_No', 'TechSupport_No_internet_service', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_No_internet_service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No_internet_service',
        'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One_year',
        'Contract_Two_year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank_transfer_(automatic)',
        'PaymentMethod_Credit_card_(automatic)',
        'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check'
    ]

    # Scaling untuk fitur numerik
    scaler = MinMaxScaler()
    scaler.fit(pd.DataFrame({
        'tenure': [0, 72],
        'MonthlyCharges': [0, 118.75],
        'TotalCharges': [0, 8684.8]
    }))

    # Fungsi untuk melakukan one-hot encoding dan normalisasi Min-Max pada input pengguna
    def preprocess_input(data):
        df = pd.DataFrame([data])

        # Normalisasi fitur numerik
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[numeric_features] = scaler.transform(df[numeric_features])

        # One-hot encoding fitur kategorikal
        df = pd.get_dummies(df)

        # Menambahkan kolom yang hilang setelah one-hot encoding
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Mengurutkan kolom sesuai model
        df = df[feature_columns]

        return df

    colored_header(
        label="Churn Prediction",
        description="",
        color_name="orange-70",
    )
    
    enter()

    tenure_col, monthly_charges_col, total_charges_col = st.columns(3)
    with tenure_col:
        tenure = st.number_input("Tenure", min_value=0, value=0)
    with monthly_charges_col:
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
    with total_charges_col:
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0)

    obj_col_1, obj_col_2, obj_col_3, obj_col_4 = st.columns(4)
    with obj_col_1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with obj_col_2:
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with obj_col_3:
        Partner = st.selectbox("Partner", ["No", "Yes"])
    with obj_col_4:
        Dependents = st.selectbox("Dependents", ["No", "Yes"])

    obj_col_5, obj_col_6, obj_col_7, obj_col_8 = st.columns(4)
    with obj_col_5:
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    with obj_col_6:
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with obj_col_7:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with obj_col_8:
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    obj_col_9, obj_col_10, obj_col_11, obj_col_12 = st.columns(4)
    with obj_col_9:
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    with obj_col_10:
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with obj_col_11:
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with obj_col_12:
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    obj_col_13, obj_col_14, obj_col_15, obj_col_16 = st.columns(4)
    with obj_col_13:
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    with obj_col_14:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with obj_col_15:
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    with obj_col_16:
        PaymentMethod = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check"
        ])

    input_data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod
    }

    # Encoding & scaling pada data yg diinput
    input_df = preprocess_input(input_data)

    # Prediksi menggunakan model
    prediction = model.predict(input_df)[0]

    enter() 
    

    if st.button("Predict"):
        if prediction == 1:
            st.success("Customer :red[akan churn].")
        else:
            st.success("Customer :green[tidak akan churn].")
            
if selected_option_menu == "Dataset Preview":
    
    colored_header(
        label="Dataset Preview",
        description="",
        color_name="orange-70",
    )
    
    enter()

    df = pd.read_csv('preview_data.csv')  
    
    _, col_filter, _ = st.columns([1,5,1])
    
    with col_filter:
        selected_features = st.multiselect('Select Features', df.columns.tolist(), default=['customerID', 'gender', 'PaymentMethod', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges'])
    
    enter()

    df = df[selected_features]
    st.dataframe(df)


