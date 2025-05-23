import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank-full.csv", sep=';')  # adjust path if needed
    return df

data = load_data()

# --------- PAGE FUNCTIONS ----------

def home():
    st.title("🏠 Term Deposit Subscription Prediction")
    st.markdown("""
    A Portuguese bank wants to improve its telemarketing campaigns by predicting which clients will subscribe to a term deposit.

    ### 💼 Business Objectives:
    - Increase campaign efficiency through predictive targeting
    - Reduce telemarketing costs
    - Improve customer experience

    Navigate through the sidebar to explore the data, make predictions, and review model performance.
    """)

# eda_page.py

def eda():
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Target Variable Distribution")
    st.bar_chart(data['y'].value_counts())

    st.subheader("Subscription by Job")
    fig, ax = plt.subplots()
    sns.countplot(x='job', hue='y', data=data, order=data['job'].value_counts().index, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap (Numerical Features)")
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)




def model_info():
    st.title("📈 Model Performance Summary")

    st.markdown("### ✅ Summary of Evaluation Metrics")
    
    # You can adjust these values with your actual results
    metrics = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
        "Accuracy": [0.89, 0.87, 0.91, 0.92, 0.915],
        "Precision": [0.86, 0.84, 0.89, 0.91, 0.883],
        "Recall": [0.84, 0.83, 0.86, 0.88, 0.857],
        "F1 Score": [0.85, 0.835, 0.875, 0.895, 0.869],
        "ROC AUC": [0.90, 0.89, 0.94, 0.95, 0.94]
    }

    import pandas as pd
    df_metrics = pd.DataFrame(metrics)
    st.dataframe(df_metrics.style.highlight_max(axis=0))

    st.markdown("### 🧩 Confusion Matrix (Example for LightGBM)")
    fig, ax = plt.subplots()
    sns.heatmap([[3600, 200], [250, 900]], annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.image("roc_logreg.png", caption="ROC Curve - Logistic Regression", use_container_width=True)
    st.image("roc_dt.png", caption="ROC Curve - Decision Tree", use_container_width=True)
    st.image("roc_rf.png", caption="ROC Curve - Random Forest", use_container_width=True)
    st.image("roc_xgb.png", caption="ROC Curve - XGBoost", use_container_width=True)
    st.image("roc_lgbm.png", caption="ROC Curve - LightGBM", use_container_width=True)


    st.markdown("### 📌 Notes")
    st.markdown("""
    - All models were evaluated using accuracy, precision, recall, F1-score, and ROC AUC.
    - ROC Curves above show model discrimination performance.
    - LightGBM and XGBoost achieved the best ROC AUC.
    """)


# --------- SIDEBAR NAVIGATION ----------

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Model Info"])

if page == "Home":
    home()
elif page == "EDA":
    eda()
elif page == "Prediction":
    import streamlit as st
    import numpy as np
    import joblib
    import time

    # Load the LightGBM model and scaler
    model = joblib.load("lightgbm_model.pkl")
    scaler = joblib.load("scaler.joblib")

    # Title and description
    st.title("Term Deposit Subscription Predictor 💼📊")
    st.write("Enter client information to predict if they'll subscribe to a term deposit.")

# Custom CSS for advanced styling with black background and improved font styling
    st.markdown("""
        <style>
        body {
            background-color: #121212; /* Dark background */
            color: #f5f5f5; /* Light text for readability */
            font-family: 'Arial', sans-serif;
            font-size: 16px;
        }

        h1 {
            font-size: 40px;
            color: #4CAF50; /* Green title */
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }

        h2 {
            font-size: 24px;
            color: #f5f5f5;
            font-weight: bold;
            margin-top: 20px;
        }

        /* Section Styling */
        .section-box {
            background-color: #1f1f1f;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        }

        .section-header {
            font-size: 22px;
            color: #4CAF50;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }

        /* Input Fields */
        .stTextInput, .stNumberInput, .stSlider, .stSelectbox, .stRadio {
            background-color: #333333;
            color: #f5f5f5;
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #555555;
            margin-bottom: 20px;
            width: 100%;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 8px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        .stTextInput>label, .stNumberInput>label, .stSlider>label, .stSelectbox>label, .stRadio>label {
            color: #f5f5f5 !important;
            font-size: 18px;
        }

        .stTextInput>input, .stNumberInput>input, .stSlider>input {
            background-color: #333333;
            color: #f5f5f5;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #555555;
        }

        /* Sidebar radio button text visibility fix */
        section[data-testid="stSidebar"] .css-17eq0hr, 
        section[data-testid="stSidebar"] .css-1v3fvcr {
            color: white !important;
            font-weight: 600;
            font-size: 16px;
        }

        </style>
    """, unsafe_allow_html=True)




    # Feature descriptions
    feature_descriptions = {
        'age': "Client's age",
        'job': "Type of job",
        'marital': "Marital status",
        'education': "Client's education level",
        'default': "Has previously defaulted on credit",
        'balance': "Average yearly account balance (€)",
        'housing': "Has a housing loan",
        'loan': "Has a personal loan",
        'contact': "Contact communication type",
        'day': "Day of the last contact (in current campaign)",
        'month': "Month of the last contact",
        'duration': "Duration of the last contact (in seconds)",
        'campaign': "Number of contacts during current campaign",
        'pdays': "Days since last contact from a previous campaign",
        'previous': "Number of contacts before this campaign",
        'poutcome': "Outcome of the previous campaign"
    }

    # Categorical Mappings
    job_mapping = {
        'management': 1, 'technician': 2, 'entrepreneur': 3, 'blue-collar': 4, 'others': 5,
        'retired': 6, 'admin': 7, 'services': 8, 'self-employed': 9, 'unemployed': 10,
        'housemaid': 11, 'student': 12
    }
    marital_mapping = {'married': 1, 'single': 2, 'divorced': 3}
    education_mapping = {'primary': 1, 'secondary': 2, 'higher secondary': 3}
    loan_default_mapping = {'no': 0, 'yes': 1}
    housing_mapping = {'yes': 1, 'no': 0}
    personal_loan_mapping = {'no': 0, 'yes': 1}
    contact_mapping = {'cellular': 1, 'telephone': 2}
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    poutcome_mapping = {'unknown': 0, 'failure': 1, 'success': 2}

    # --- Form Inputs ---

    st.markdown('<div class="section-header">Client\'s Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider(feature_descriptions['age'], 18, 100, 30)
        job = st.selectbox(feature_descriptions['job'], list(job_mapping.keys()))
        marital = st.selectbox(feature_descriptions['marital'], list(marital_mapping.keys()))
    with col2:
        education = st.selectbox(feature_descriptions['education'], list(education_mapping.keys()))
        default = st.radio(feature_descriptions['default'], list(loan_default_mapping.keys()))
        balance = st.number_input(feature_descriptions['balance'], -5000, 100000, 1000)

    st.markdown('<div class="section-header">Financial Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        housing = st.radio(feature_descriptions['housing'], list(housing_mapping.keys()))
        loan = st.radio(feature_descriptions['loan'], list(personal_loan_mapping.keys()))
        contact = st.selectbox(feature_descriptions['contact'], list(contact_mapping.keys()))
    with col2:
        day = st.slider(feature_descriptions['day'], 1, 31, 15)
        month = st.selectbox(feature_descriptions['month'], list(month_mapping.keys()))
        duration = st.slider(feature_descriptions['duration'], 0, 3000, 100)

    st.markdown('<div class="section-header">Previous Contacts</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        campaign = st.slider(feature_descriptions['campaign'], 1, 50, 5)
        pdays = st.slider(feature_descriptions['pdays'], -1, 999, -1)
    with col2:
        previous = st.slider(feature_descriptions['previous'], 0, 50, 0)
        poutcome = st.selectbox(feature_descriptions['poutcome'], list(poutcome_mapping.keys()))

    # Convert inputs to numeric values
    features = np.array([[
        age,
        job_mapping[job],
        marital_mapping[marital],
        education_mapping[education],
        loan_default_mapping[default],
        balance,
        housing_mapping[housing],
        personal_loan_mapping[loan],
        contact_mapping[contact],
        day,
        month_mapping[month],
        duration,
        campaign,
        pdays,
        previous,
        poutcome_mapping[poutcome]
    ]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    if st.button("Predict", use_container_width=True):
        with st.spinner("Making prediction..."):
            time.sleep(2)
            prediction = model.predict(scaled_features)[0]
            result = "✅ Subscribed to Term Deposit" if prediction == 1 else "❌ Not Subscribed"
            st.success(f"Prediction: {result}")

elif page == "Model Info":
    model_info()
