import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Predictive Machine Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for Industrial Background Image with 40% Opacity Overlay
st.markdown("""
    <style>
    /* Main App Background - Industrial Machinery Image with 40% Blue Overlay */
    .stApp {
        background-image: linear-gradient(rgba(219, 234, 254, 0.40), rgba(219, 234, 254, 0.40)), 
                          url("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #1e3a8a;
    }
    
    /* Sidebar Background Color - Frosted Glass Effect */
    [data-testid="stSidebar"] {
        background-color: rgba(191, 219, 254, 0.90);
        border-right: 1px solid rgba(147, 197, 253, 0.6);
    }

    /* Top Header Background */
    header[data-testid="stHeader"] {
        background-color: rgba(219, 234, 254, 0.3);
    }

    /* Metric Card Styling - Frosted White Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(147, 197, 253, 0.9);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.15);
        text-align: center;
    }
    .metric-title {
        font-size: 13px;
        color: #2563eb;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        color: #1d4ed8;
        font-weight: bold;
    }

    /* Content Cards for Visibility */
    .info-card {
        background: rgba(255, 255, 255, 0.92);
        border-left: 5px solid #2563eb;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .conclusion-card {
        background: rgba(255, 255, 255, 0.92);
        border-left: 5px solid #059669;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Customizing Tabs Background and Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(191, 219, 254, 0.85);
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1e40af;
        font-weight: 600;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.98) !important;
        color: #1d4ed8 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title & Header (Font sizes left completely unmodified)
st.title("⚙️ Predictive Machine Maintenance Dashboard")
st.markdown("##### Real-time Machine Health Simulation & Transition Analysis via Markov Chains (Experiment 07)")
st.divider()

# --- ALWAYS VISIBLE OBJECTIVE & PROBLEM STATEMENT ---
st.markdown("""
    <div class="info-card">
        <h4 style="margin-top:0; color:#1e3a8a;">🎯 Objective & Problem Statement</h4>
        <p style="margin-bottom:0;">
            To <b>predict machine health states and failure probability</b> using a <b>Markov Process</b>, 
            so that preventive maintenance can be scheduled proactively to reduce downtime. Manufacturing machines gradually degrade over time; modeling this as a Markov chain allows us to forecast breakdowns and optimize intervention intervals.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🛠️ Simulation Controls")
start_state_name = st.sidebar.selectbox(
    "Initial Machine State",
    ["Operational", "Minor Wear", "Major Fault", "Failure"],
    index=0
)
n_steps = st.sidebar.slider("Simulation Time Steps", min_value=5, max_value=50, value=20, step=1)

state_mapping = {"Operational": 0, "Minor Wear": 1, "Major Fault": 2, "Failure": 3}
states = ["Operational", "Minor Wear", "Major Fault", "Failure"]
start_state = state_mapping[start_state_name]

# Transition Probability Matrix P
P = np.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.2, 0.5, 0.2, 0.1],
    [0.0, 0.3, 0.4, 0.3],
    [0.0, 0.0, 0.0, 1.0]
])

# Run Simulation
history = []
current_state = start_state
for _ in range(n_steps):
    history.append(current_state)
    current_state = np.random.choice([0, 1, 2, 3], p=P[current_state])

final_state_name = states[history[-1]]
failure_occurred = "Yes" if 3 in history else "No"

# Top Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Initial State</div>
            <div class="metric-value" style="color: #059669;">{start_state_name}</div>
        </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Final State</div>
            <div class="metric-value" style="color: {'#dc2626' if final_state_name=='Failure' else '#d97706'};">{final_state_name}</div>
        </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Steps</div>
            <div class="metric-value">{n_steps}</div>
        </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Failure Triggered</div>
            <div class="metric-value" style="color: {'#dc2626' if failure_occurred=='Yes' else '#059669'};">{failure_occurred}</div>
        </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# Main Dashboard Sections / Tabs
tab1, tab2 = st.tabs(["📈 Simulation & Visualization", "📊 Matrix, Setup & Conclusion"])

with tab1:
    col_sim1, col_sim2 = st.columns([1, 1.2])
    
    with col_sim1:
        st.subheader("📋 State Transition Log")
        sim_data = [{"Step": i+1, "State": states[s]} for i, s in enumerate(history)]
        df_sim = pd.DataFrame(sim_data)
        
        # Pastel Color-coding function for table cells
        def highlight_states(val):
            color_map = {
                "Operational": "background-color: #d1fae5; color: #065f46; font-weight: bold;",
                "Minor Wear": "background-color: #fef3c7; color: #92400e; font-weight: bold;",
                "Major Fault": "background-color: #ffedd5; color: #9a3412; font-weight: bold;",
                "Failure": "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
            }
            return color_map.get(val, "")

        st.dataframe(df_sim.style.applymap(highlight_states, subset=['State']), height=380, use_container_width=True)
        
    with col_sim2:
        st.subheader("📉 Machine Degradation Trend")
        
        # Matplotlib Graph with transparent background matching theme
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor('#dbeafe')
        fig.patch.set_alpha(0.7)              
        ax.set_facecolor('#ffffff')         
        
        ax.plot(range(1, n_steps + 1), history, marker='o', linestyle='-', color='#2563eb', linewidth=2.5, markersize=6)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(states)
        ax.set_title("Machine Health State Over Time", color='#1e3a8a', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Time Step", color='#1e40af')
        ax.set_ylabel("Health State", color='#1e40af')
        ax.grid(True, linestyle='--', alpha=0.5, color='#93c5fd')
        
        # Clean borders
        for spine in ax.spines.values():
            spine.set_color('#93c5fd')
            
        st.pyplot(fig)

with tab2:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("🏷️ Machine Health States")
        state_df = pd.DataFrame({
            "State ID": ["S1", "S2", "S3", "S4"],
            "Condition": ["Fully Operational", "Minor Wear", "Major Fault", "Machine Failure"]
        })
        st.table(state_df)
    
    with col_t2:
        st.subheader("🔢 Transition Probability Matrix (P)")
        matrix_df = pd.DataFrame(
            P,
            index=["S1 (Operational)", "S2 (Minor Wear)", "S3 (Major Fault)", "S4 (Failure)"],
            columns=["S1", "S2", "S3", "S4"]
        )
        st.dataframe(matrix_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    # --- CONCLUSION LOCATED IN 2ND TAB ---
    st.markdown("""
        <div class="conclusion-card">
            <h4 style="margin-top:0; color:#065f46;">📝 Conclusion</h4>
            <p style="margin-bottom:0;">
                The Markov process effectively models manufacturing wear-and-tear dynamics. By analyzing these 
                transition probabilities, industrial plants can forecast potential failures early, maximize machine reliability, 
                and minimize unexpected downtime.
            </p>
        </div>
    """, unsafe_allow_html=True)