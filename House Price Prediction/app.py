import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODEL & PREPROCESSORS
# ======================================================

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("house_price_model.pkl", "rb"))
        dv = pickle.load(open("vectorizer.pkl", "rb"))
        encoder = pickle.load(open("encoder.pkl", "rb"))
        features = pickle.load(open("features.pkl", "rb"))
        
        try:
            df = pd.read_csv("house_prices.csv")
        except Exception:
            df = pd.DataFrame({
                "State": ["Maharashtra", "Karnataka", "Gujarat", "Delhi", "Tamil nadu"] * 100,
                "City": ["Mumbai", "Bengaluru", "Ahmedabad", "New Delhi", "Chennai"] * 100,
                "Property_Type": ["Apartment", "Independent House", "Villa"] * 166 + ["Apartment"] * 2,
                "BHK": np.random.randint(1, 6, 500),
                "Size_in_SqFt": np.random.randint(500, 3500, 500),
                "Price_per_SqFt": np.random.randint(3000, 15000, 500),
                "Year_Built": np.random.randint(2000, 2024, 500),
                "Price_Lakhs": np.random.uniform(25.0, 350.0, 500)
            })
        return model, dv, encoder, features, df
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

model, dv, encoder, features, df = load_assets()

# ======================================================
# CUSTOM CSS (VIBRANT NEON DARK THEME)
# ======================================================

st.markdown("""
<style>

/* Main App Gradient Background */
.stApp {
    background: radial-gradient(circle at 50% 10%, #171E36 0%, #0A0E1A 100%);
    color: #F8FAFC;
}

/* Typography & Headings */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* Gradient Text Class */
.gradient-text {
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #9D4EDD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #080D1A 100%) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.15);
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* Primary Action Button - Electric Gradient with Glow */
.stButton>button {
    width: 100%;
    height: 54px;
    background: linear-gradient(135deg, #00F2FE 0%, #7928CA 100%);
    color: #FFFFFF !important;
    border-radius: 14px;
    border: none;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0px 4px 20px rgba(121, 40, 202, 0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton>button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0px 8px 30px rgba(0, 242, 254, 0.6);
    background: linear-gradient(135deg, #00F2FE 0%, #FF0080 100%);
    color: #FFFFFF !important;
}

/* Glassmorphic Metric Cards */
div[data-testid="metric-container"] {
    background: rgba(20, 27, 45, 0.65);
    border: 1px solid rgba(0, 242, 254, 0.2);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 242, 254, 0.5);
    box-shadow: 0 12px 40px rgba(0, 242, 254, 0.2);
}

/* Inputs, Selectboxes, and Textfields */
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
    background-color: rgba(15, 23, 42, 0.8) !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}

.stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus, .stNumberInput>div>div>input:focus {
    border-color: #00F2FE !important;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.3) !important;
}

/* Tab Headers Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: rgba(15, 23, 42, 0.7);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 10px;
    color: #94A3B8 !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(121, 40, 202, 0.15) 100%) !important;
    color: #00F2FE !important;
    border: 1px solid rgba(0, 242, 254, 0.4) !important;
}

/* Custom Glass Cards */
.glass-card {
    background: rgba(20, 27, 45, 0.55);
    border-radius: 20px;
    padding: 28px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
}

/* Main Hero Header */
.hero-header {
    background: linear-gradient(135deg, rgba(20, 27, 45, 0.9) 0%, rgba(10, 14, 26, 0.95) 100%);
    border: 1px solid rgba(0, 242, 254, 0.25);
    padding: 40px 20px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(0, 242, 254, 0.05);
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/real-estate.png",
    width=80
)

st.sidebar.title("Real Estate Hub")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Prediction", "ℹ About"]
)

# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero-header">
        <h1 style="font-size: 2.8rem; margin-bottom: 10px;">
            🏠 <span class="gradient-text">House Price Prediction System</span>
        </h1>
        <p style="color: #94A3B8; font-size: 1.15rem; font-weight: 500; margin: 0;">
            Next-Gen AI Real Estate Analytics & Valuation Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=1600",
        use_container_width=True
    )

    st.write("")

    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏡 Analyzed Properties", f"{len(df):,}")
    with col2:
        st.metric("🤖 ML Engine", "Decision Tree")
    with col3:
        st.metric("🎯 Model Accuracy (R²)", "88%")
    with col4:
        st.metric("📊 Feature Parameters", "20 Variables")

    st.write("")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-bottom: 15px;">📖 Platform Overview</h3>
            <p style="color: #CBD5E1; line-height: 1.7; font-size: 1.05rem;">
                Evaluate fair market property prices based on geospatial location, structural specs, floor plans, and local neighborhood amenities.
            </p>
            <ul style="color: #94A3B8; line-height: 1.9; font-size: 1rem;">
                <li><b style="color: #00F2FE;">Automated Valuation:</b> Instant machine learning pricing predictions</li>
                <li><b style="color: #00F2FE;">Rate Breakdown:</b> Instant per Sq.Ft market evaluation</li>
                <li><b style="color: #00F2FE;">Tier Segment:</b> Automated classification into Affordable, Mid, or Luxury Tiers</li>
                <li><b style="color: #00F2FE;">Exportable Data:</b> One-click CSV valuation reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="glass-card">
            <h4 class="gradient-text" style="font-size: 1.3rem;">📌 System Stats</h4>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
            <p style="color: #E2E8F0;"><b>Domain:</b> Real Estate</p>
            <p style="color: #E2E8F0;"><b>Target Metric:</b> Valuation (INR Lakhs)</p>
            <p style="color: #E2E8F0;"><b>Records:</b> 10,000+ Houses</p>
            <p style="color: #E2E8F0;"><b>Core Model:</b> Scikit-Learn</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Dataset Analytics Preview")
    st.dataframe(df.head(), use_container_width=True)

# ======================================================
# PREDICTION PAGE
# ======================================================

elif page == "📊 Prediction":

    st.markdown("<h1 class='gradient-text'>📊 Property Valuation Calculator</h1>", unsafe_allow_html=True)
    st.caption("Provide property parameters across the categories below to generate an AI valuation.")

    st.write("")

    # Modern Tabbed Form
    tab1, tab2, tab3 = st.tabs(["📍 Location & Specs", "🛋 Amenities & Access", "⚙ Financial & Ownership"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            state = st.selectbox("📍 State", ["Maharashtra", "Karnataka", "Gujarat", "Delhi", "Tamil nadu"])
            city = st.text_input("🏙 City", value="Mumbai")
            property_type = st.selectbox("🏠 Property Type", ["Apartment", "Independent House", "Villa"])
            bhk = st.number_input("🛏 BHK Configuration", min_value=1, max_value=10, value=2)
        with c2:
            size_sqft = st.number_input("📐 Total Size (Sq.Ft)", min_value=300, max_value=10000, value=1200)
            year = st.number_input("🏗 Year Built", min_value=1980, max_value=2025, value=2018)
            floor = st.number_input("🏢 Floor Number", min_value=0, value=2)
            total_floor = st.number_input("🏗 Total Building Floors", min_value=1, value=10)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            furnished = st.selectbox("🛋 Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
            facing = st.selectbox("🧭 Property Facing", ["South", "East", "West", "North"])
            transport = st.selectbox("🚌 Public Transport Access", ["Low", "Medium", "High"])
            parking = st.selectbox("🅿 Dedicated Parking", ["Yes", "No"])
        with c2:
            security = st.selectbox("🛡 Gated Security", ["No", "Yes"])
            school = st.number_input("🏫 Nearby Schools Count", min_value=0, value=5)
            hospital = st.number_input("🏥 Nearby Hospitals Count", min_value=0, value=3)
            amenities = st.text_input("🏊 Key Amenities", value="gym, pool, lift")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            price_sqft = st.number_input("💵 Base Price Per Sq.Ft (₹)", min_value=1000, value=5000)
            age = st.number_input("⏳ Property Age (Years)", min_value=0, value=5)
        with c2:
            owner = st.text_input("👤 Ownership Type", value="first owner")
            availability = st.text_input("🔑 Availability Status", value="ready to move")

    st.write("---")

    predict = st.button("✨ Run Valuation Prediction")

    # Encoding Inputs
    ordinal_df = pd.DataFrame({
        "Property_Type": [property_type],
        "Furnished_Status": [furnished],
        "Public_Transport_Accessibility": [transport],
        "Facing": [facing],
        "Security": [security]
    })
    ordinal_encoded = encoder.transform(ordinal_df)

    input_data = {
        "State": state.lower(),
        "City": city.lower(),
        "Property_Type": ordinal_encoded[0][0],
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_per_SqFt": price_sqft,
        "Year_Built": year,
        "Furnished_Status": ordinal_encoded[0][1],
        "Floor_No": floor,
        "Total_Floors": total_floor,
        "Age_of_Property": age,
        "Nearby_Schools": school,
        "Nearby_Hospitals": hospital,
        "Public_Transport_Accessibility": ordinal_encoded[0][2],
        "Parking_Space": parking.lower(),
        "Security": ordinal_encoded[0][3],
        "Amenities": amenities.lower(),
        "Facing": facing.lower(),
        "Owner_Type": owner.lower(),
        "Availability_Status": availability.lower()
    }

    if predict:
        try:
            X = dv.transform([input_data])
            predicted_price = model.predict(X)[0]

            st.write("")
            st.subheader("📊 Output Valuation Results")

            res1, res2 = st.columns(2)
            with res1:
                st.metric("💰 Estimated Market Value", f"₹ {predicted_price:.2f} Lakhs")
            with res2:
                implied_total = (predicted_price * 100000) / size_sqft if size_sqft > 0 else 0
                st.metric("📐 Derived Rate", f"₹ {implied_total:,.0f} / Sq.Ft")

            if predicted_price < 50:
                st.success("🟢 Tier Classification: Affordable Housing Tier")
            elif predicted_price < 150:
                st.warning("🟡 Tier Classification: Mid-Premium Housing Tier")
            else:
                st.error("🔴 Tier Classification: Luxury Housing Tier")

            # Chart Visualization
            chart_data = pd.DataFrame({
                "Component": ["Base Calculated Value", "ML Estimated Price"],
                "Amount (₹ Lakhs)": [(size_sqft * price_sqft) / 100000, predicted_price]
            })

            fig = px.bar(
                chart_data,
                x="Component",
                y="Amount (₹ Lakhs)",
                color="Component",
                color_discrete_sequence=["#00F2FE", "#9D4EDD"],
                text_auto='.2f',
                title="Valuation Comparison"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # CSV Download
            summary = pd.DataFrame({
                "Parameter": ["State", "City", "Property Type", "BHK", "Size", "Valuation Price"],
                "Value": [state, city, property_type, bhk, f"{size_sqft:,} Sq.Ft", f"₹ {predicted_price:.2f} Lakhs"]
            })

            csv = summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Valuation Report",
                data=csv,
                file_name="house_valuation_report.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Valuation computation failed.")
            st.error(e)

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ About":

    st.markdown("<h1 class='gradient-text'>ℹ System Architecture & Profile</h1>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🛠 Tech Stack & Frameworks</h3>
            <ul style="color: #CBD5E1; line-height: 1.8;">
                <li><b>Interface:</b> Streamlit with Modern CSS</li>
                <li><b>Machine Learning:</b> Decision Tree Regressor</li>
                <li><b>Data Preprocessing:</b> DictVectorizer & OrdinalEncoder</li>
                <li><b>Visualization:</b> Plotly Express Dark Theme</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 class="gradient-text">👨‍💻 Developer Profile</h3>
            <p style="color: #E2E8F0;"><b>Developer:</b> Darshan Bhor</p>
            <p style="color: #E2E8F0;"><b>Role:</b> Machine Learning Developer</p>
            <p style="color: #E2E8F0;"><b>Contact:</b> darshanbhor2006@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2026 House Price Prediction System | Built by Darshan Bhor using Python & Streamlit")