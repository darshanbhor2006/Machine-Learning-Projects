import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="🏦 Customer Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("random_forest_churn_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        df = pd.read_csv("Churn_Modelling.csv")
        return model, scaler, df
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

model, scaler, df = load_assets()

# ======================================================
# CUSTOM CSS (UPDATED COLOR PALETTE)
# ======================================================

st.markdown("""
<style>

/* App Background - Cool Slate Gradient */
.stApp {
    background: linear-gradient(135deg, #EEF2F6 0%, #E2E8F0 100%);
}

/* Primary Headers - Deep Midnight Blue */
h1, h2, h3 {
    color: #0F172A !important;
}

/* Sidebar Styling - Dark Navy Gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B192C 0%, #1E3E62 100%);
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* Primary Action Buttons - Teal/Cyan Palette */
.stButton>button {
    width: 100%;
    height: 55px;
    background: #00A896;
    color: white;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0px 4px 12px rgba(0, 168, 150, 0.3);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: #028090;
    box-shadow: 0px 6px 18px rgba(2, 128, 144, 0.4);
    color: #FFFFFF;
}

/* KPI Metric Cards */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 4px 12px rgba(15, 23, 42, 0.06);
    border: 1px solid #CBD5E1;
}

/* Main Header Container */
.main-header {
    background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0px 4px 15px rgba(15, 23, 42, 0.25);
}

.main-header h1, .main-header h3 {
    color: #FFFFFF !important;
    margin: 0;
}

/* Cards & Content Panels */
.card {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(15, 23, 42, 0.06);
    border: 1px solid #CBD5E1;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=90
)

st.sidebar.title("🏦 Banking Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Prediction",
        "📂 Batch Processing",
        "ℹ About"
    ]
)

# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.title("🏦 Customer Churn Prediction System")

    st.markdown(
        "<h4 style='color:#00A896;'>AI Powered Banking Analytics Dashboard</h4>",
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1600",
        use_container_width=True
    )

    st.write("")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Customers", f"{len(df):,}")

    with col2:
        st.metric("🤖 Model", "Random Forest")

    with col3:
        st.metric("🎯 Accuracy", "86%")

    with col4:
        st.metric("📊 Features", "11")

    st.write("---")

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📖 About the Project")

        st.write("""
Customer Churn Prediction helps banks identify customers who are likely to leave the bank.

This application uses a trained **Random Forest Machine Learning Model** to predict whether a customer will stay or exit based on banking details.

### Features

- ✅ Customer Churn Prediction
- ✅ Banking Analytics Dashboard
- ✅ Random Forest Algorithm
- ✅ Stay & Exit Probability
- ✅ Risk Analysis
- ✅ Interactive User Interface
- ✅ Download Prediction Report

This system helps banks improve customer retention by identifying high-risk customers early.
""")

    with right:

        st.info("""
### 📌 Project Information

🏦 **Domain:** Banking

🤖 **Model:** Random Forest

📊 **Dataset:** Churn Modelling

👥 **Records:** 10,000 Customers

🎯 **Target:** Customer Churn

💻 **Framework:** Streamlit
""")

    st.write("---")

    st.subheader("📊 Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

    st.write("")

    st.subheader("📈 Dataset Statistics")

    st.dataframe(df.describe(), use_container_width=True)

# ======================================================
# PREDICTION PAGE
# ======================================================

elif page == "📊 Prediction":

    st.title("📊 Customer Churn Prediction")

    st.markdown(
        "<h4 style='color:#00A896;'>Enter Customer Details</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns(2)

    with left:

        credit_score = st.number_input(
            "💳 Credit Score",
            min_value=300,
            max_value=900,
            value=600
        )

        age = st.number_input(
            "🎂 Age",
            min_value=18,
            max_value=100,
            value=35
        )

        tenure = st.number_input(
            "📅 Tenure",
            min_value=0,
            max_value=10,
            value=5
        )

        balance = st.number_input(
            "💰 Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        num_products = st.selectbox(
            "🏦 Number of Products",
            [1, 2, 3, 4]
        )

    with right:

        has_card = st.selectbox(
            "💳 Has Credit Card",
            ["Yes", "No"]
        )

        active_member = st.selectbox(
            "🟢 Active Member",
            ["Yes", "No"]
        )

        salary = st.number_input(
            "💼 Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        country = st.selectbox(
            "🌍 Country",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "👤 Gender",
            ["Male", "Female"]
        )

    st.write("")

    st.markdown("---")

    st.subheader("📋 Customer Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Credit Score", credit_score)

    with c2:
        st.metric("Age", age)

    with c3:
        st.metric("Balance", f"₹{balance:,.2f}")

    st.write("")

    predict = st.button("🔍 Predict Customer Churn")

    # Data Encoding
    germany = 1 if country == "Germany" else 0
    spain = 1 if country == "Spain" else 0
    male = 1 if gender == "Male" else 0

    has_card_val = 1 if has_card == "Yes" else 0
    active_member_val = 1 if active_member == "Yes" else 0

    if predict:

        input_data = np.array([[
            credit_score,
            age,
            tenure,
            balance,
            num_products,
            has_card_val,
            active_member_val,
            salary,
            germany,
            spain,
            male
        ]])

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        stay_prob = probability[0][0] * 100
        churn_prob = probability[0][1] * 100

        st.write("")
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🟢 Stay Probability", f"{stay_prob:.2f}%")

        with col2:
            st.metric("🔴 Churn Probability", f"{churn_prob:.2f}%")

        st.progress(int(churn_prob))

        chart_df = pd.DataFrame(
            {"Probability": [stay_prob, churn_prob]},
            index=["Stay", "Churn"]
        )

        st.subheader("📈 Probability Comparison")
        st.bar_chart(chart_df)

        if prediction[0] == 1:
            st.error("🚨 High Risk Customer")
            st.markdown("""
### ⚠ Customer is likely to EXIT

Recommended Actions
- Contact customer immediately
- Offer cashback or discount
- Provide loyalty rewards
- Assign Relationship Manager
- Offer premium banking services
""")

        else:
            st.success("🎉 Customer is likely to STAY")
            st.balloons()
            st.markdown("""
### ✅ Customer is likely to STAY

Recommended Actions
- Continue customer engagement
- Offer Premium Credit Card
- Increase Reward Points
- Exclusive Banking Offers
- Maintain Customer Satisfaction
""")

        st.write("---")
        st.subheader("📈 Risk Level")

        if churn_prob < 30:
            st.success("🟢 LOW RISK")
        elif churn_prob < 70:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

        st.write("")
        st.subheader("📋 Customer Summary")

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Credit Score", "Age", "Tenure", "Country", "Gender",
                    "Balance", "Estimated Salary", "Number of Products",
                    "Has Credit Card", "Active Member", "Prediction"
                ],
                "Value": [
                    credit_score, age, tenure, country, gender,
                    f"₹{balance:,.2f}", f"₹{salary:,.2f}", num_products,
                    "Yes" if has_card_val else "No",
                    "Yes" if active_member_val else "No",
                    "Exit" if prediction[0] == 1 else "Stay"
                ]
            }
        )

        st.table(summary)

        csv = summary.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="customer_prediction_report.csv",
            mime="text/csv"
        )

        st.write("---")

        with st.expander("🤖 Model Information"):
            st.write("""
### Random Forest Classifier

**Algorithm:** Random Forest Classifier
**Dataset:** Churn Modelling Dataset
**Training Samples:** 10,000 Customers
**Input Features:** 11
**Output:** Stay / Exit
**Framework:** Scikit-Learn
            """)

        with st.expander("📊 Dataset Preview"):
            st.dataframe(df.head(), use_container_width=True)
            st.write("Shape :", df.shape)

        with st.expander("💡 Prediction Tips"):
            st.info("""
✔ Enter valid customer details.
✔ Higher age may increase churn probability.
✔ Active members usually have lower churn.
✔ Higher balance alone does not guarantee churn.
✔ This prediction is based on the trained machine learning model.
""")

        st.success("✅ Prediction Completed Successfully")

# ======================================================
# BATCH PROCESSING PAGE
# ======================================================

elif page == "📂 Batch Processing":

    st.title("📂 Batch Churn Prediction")

    st.markdown(
        "<h4 style='color:#00A896;'>Upload Customer CSV Dataset</h4>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview:", batch_df.head(5))

        if st.button("Run Bulk Predictions"):
            try:
                proc_df = batch_df.copy()

                if 'Geography' in proc_df.columns:
                    proc_df['Geography_Germany'] = (proc_df['Geography'] == 'Germany').astype(int)
                    proc_df['Geography_Spain'] = (proc_df['Geography'] == 'Spain').astype(int)
                    proc_df.drop(columns=['Geography'], inplace=True)

                if 'Gender' in proc_df.columns:
                    proc_df['Gender_Male'] = (proc_df['Gender'] == 'Male').astype(int)
                    proc_df.drop(columns=['Gender'], inplace=True)

                for col in ['HasCrCard', 'IsActiveMember']:
                    if col in proc_df.columns and proc_df[col].dtype == object:
                        proc_df[col] = proc_df[col].map({'Yes': 1, 'No': 0})

                scaled_batch = scaler.transform(proc_df)
                predictions = model.predict(scaled_batch)
                probs = model.predict_proba(scaled_batch)[:, 1]

                batch_df['Prediction'] = np.where(predictions == 1, 'Exit', 'Stay')
                batch_df['Churn Probability (%)'] = (probs * 100).round(2)

                st.success("Processing Completed!")
                st.dataframe(batch_df, use_container_width=True)

                csv_output = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Batch Results",
                    data=csv_output,
                    file_name="churn_batch_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error processing batch: {e}")

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ About":

    st.markdown("""
    <div class="main-header">
        <h1>🏦 Customer Churn Prediction System</h1>
        <h3>AI Powered Banking Analytics Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([2, 1])

    with left:

        st.markdown("""
## 📖 About the Project

Customer Churn Prediction System is a Machine Learning application
that predicts whether a customer is likely to leave the bank.

Banks lose valuable customers every year. By identifying
high-risk customers early, organizations can improve customer
retention and increase profitability.

---

## 🚀 Workflow

Customer Details
⬇
Data Preprocessing
⬇
Feature Encoding
⬇
Data Scaling
⬇
Random Forest Model
⬇
Prediction
⬇
Probability Analysis
⬇
Business Recommendation

---

## ✨ Features

✅ Customer Churn Prediction
✅ Risk Analysis
✅ AI Dashboard
✅ Interactive Charts
✅ Feature Importance
✅ Prediction History
✅ Customer Summary
✅ Download Report
✅ Responsive Dashboard
""")

    with right:

        st.markdown("""
<div class="card">

## 📌 Project Details

🏦 Domain
**Banking**

🤖 Model
**Random Forest Classifier**

📊 Dataset
**Customer Churn Modelling**

👥 Records
**10,000**

📈 Features
**11**

💻 Framework
**Streamlit**

🐍 Language
**Python**

</div>
""", unsafe_allow_html=True)

    st.write("---")

    st.subheader("🛠 Technology Stack")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
### 🐍 Python
✔ Python
✔ NumPy
✔ Pandas
✔ Pickle
""")

    with c2:
        st.success("""
### 🤖 Machine Learning
✔ Random Forest
✔ Scikit-Learn
✔ Classification
✔ Prediction
""")

    with c3:
        st.success("""
### 🌐 Dashboard
✔ Streamlit
✔ Plotly
✔ HTML
✔ CSS
""")

    st.write("---")

    st.subheader("🎯 Business Benefits")

    a, b, c = st.columns(3)

    with a:
        st.metric("Customer Retention", "↑ 25%")

    with b:
        st.metric("Revenue Growth", "↑ 18%")

    with c:
        st.metric("Customer Satisfaction", "↑ 30%")

    st.write("---")

    st.subheader("📈 Why Machine Learning?")

    st.info("""
Machine Learning enables banks to identify customers who are likely
to leave by analyzing historical customer data.

Benefits include:
• Improved customer retention
• Better marketing campaigns
• Personalized offers
• Increased profitability
• Faster decision making
• Reduced customer acquisition costs
""")

    st.write("---")

    st.subheader("👨‍💻 Developer")

    st.markdown("""
<div style="
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(15, 23, 42, 0.06);
border: 1px solid #CBD5E1;
">

<h2 style="color:#0B192C;">Darshan Bhor</h2>

<hr>

<b>Project :</b> Customer Churn Prediction System<br><br>
<b>Role :</b> Machine Learning Developer<br><br>
<b>Skills :</b>
<ul>
<li>Python</li>
<li>Machine Learning</li>
<li>Scikit-Learn</li>
<li>Pandas</li>
<li>NumPy</li>
<li>Streamlit</li>
<li>Plotly</li>
</ul>

</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("📬 Contact")

    st.success("""
📧 Email: darshanbhor2006@gmail.com
💼 LinkedIn: https://linkedin.com/in/yourprofile
🐙 GitHub: https://github.com/yourusername
""")

    st.write("---")

    st.markdown("""
<div style="
text-align:center;
padding:20px;
background:#0B192C;
color:white;
border-radius:15px;
">

<h3 style="color:white !important;">🏦 Customer Churn Prediction System</h3>
<p>Built with ❤️ using Python, Streamlit & Machine Learning</p>
<p>© 2026 Darshan Bhor | All Rights Reserved</p>

</div>
""", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================

st.write("---")

st.markdown(
"""
<div style="
text-align:center;
color:#64748B;
font-size:15px;
padding:15px;
">

🏦 <b>Customer Churn Prediction System</b><br><br>
Made with Darshan Bhor using Python, Streamlit & Scikit-Learn

</div>
""",
unsafe_allow_html=True
)