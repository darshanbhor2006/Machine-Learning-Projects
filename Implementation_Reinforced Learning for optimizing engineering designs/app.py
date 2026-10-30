import os
import pickle
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="RL Engineering Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cards, metrics, and tab layout
st.markdown("""
<style>
    /* Gradient Header Accent */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Styled Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #34d399;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "saved_models"
ENERGY_FILE = os.path.join(MODEL_DIR, "energy_optimization.pkl")
ROBOT_FILE = os.path.join(MODEL_DIR, "robot_navigation.pkl")

ENERGY_STATES = ["Low", "Medium", "High"]
ENERGY_ACTIONS = ["Reduce", "Maintain", "Increase"]

ROBOT_ACTIONS = [
    (0, 1),   # Right
    (0, -1),  # Left
    (1, 0),   # Down
    (-1, 0)   # Up
]
ROBOT_ACTION_NAMES = ["Right", "Left", "Down", "Up"]


# ---------------------------------------------------------
# Model Training Logic
# ---------------------------------------------------------

def train_energy_model():
    """Train the energy optimization Q-learning model."""
    q_table = np.zeros((len(ENERGY_STATES), len(ENERGY_ACTIONS)))

    rewards = np.array([
        [5, 2, -1],    # Low load
        [3, 4, 0],     # Medium load
        [-2, 2, 5]     # High load
    ])

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    episodes = 1000

    for _ in range(episodes):
        state = random.randint(0, 2)

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = int(np.argmax(q_table[state]))

        reward = rewards[state, action]
        next_state = random.randint(0, 2)

        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state])
            - q_table[state, action]
        )

    return {
        "Q": q_table,
        "states": ENERGY_STATES,
        "actions": ENERGY_ACTIONS
    }


def train_robot_model():
    """Train the 5 x 5 robot navigation Q-learning model."""
    grid_size = 5
    start = (0, 0)
    goal = (4, 4)

    q_table = np.zeros((grid_size, grid_size, len(ROBOT_ACTIONS)))

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    episodes = 500
    max_steps_per_episode = 100

    for _ in range(episodes):
        state = start

        for _ in range(max_steps_per_episode):
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = int(np.argmax(q_table[state[0], state[1]]))

            move = ROBOT_ACTIONS[action]
            new_state = (
                max(0, min(grid_size - 1, state[0] + move[0])),
                max(0, min(grid_size - 1, state[1] + move[1]))
            )

            reward = 100 if new_state == goal else -1

            if new_state == goal:
                target = reward
            else:
                target = reward + gamma * np.max(
                    q_table[new_state[0], new_state[1]]
                )

            q_table[state[0], state[1], action] += alpha * (
                target - q_table[state[0], state[1], action]
            )

            state = new_state
            if state == goal:
                break

    return {
        "Q": q_table,
        "grid_size": grid_size,
        "goal": goal,
        "start": start,
        "actions": ROBOT_ACTIONS,
        "action_names": ROBOT_ACTION_NAMES
    }


def save_model(path, model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_or_create_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(ENERGY_FILE):
        try:
            with open(ENERGY_FILE, "rb") as file:
                energy_model = pickle.load(file)
        except Exception:
            energy_model = train_energy_model()
            save_model(ENERGY_FILE, energy_model)
    else:
        energy_model = train_energy_model()
        save_model(ENERGY_FILE, energy_model)

    if os.path.exists(ROBOT_FILE):
        try:
            with open(ROBOT_FILE, "rb") as file:
                robot_model = pickle.load(file)
        except Exception:
            robot_model = train_robot_model()
            save_model(ROBOT_FILE, robot_model)
    else:
        robot_model = train_robot_model()
        save_model(ROBOT_FILE, robot_model)

    return energy_model, robot_model


def get_robot_path(robot_model, max_steps=50):
    q_table = np.asarray(robot_model["Q"])
    grid_size = int(robot_model.get("grid_size", 5))
    start = tuple(robot_model.get("start", (0, 0)))
    goal = tuple(robot_model.get("goal", (4, 4)))
    actions = robot_model.get("actions", ROBOT_ACTIONS)

    state = start
    path = [state]
    visited = {state}

    for _ in range(max_steps):
        if state == goal:
            break

        action_index = int(np.argmax(q_table[state[0], state[1]]))
        move = actions[action_index]

        next_state = (
            max(0, min(grid_size - 1, state[0] + move[0])),
            max(0, min(grid_size - 1, state[1] + move[1]))
        )

        if next_state == state or next_state in visited:
            break

        state = next_state
        path.append(state)
        visited.add(state)

    return path, goal, grid_size


# ---------------------------------------------------------
# Sidebar & Header
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    st.markdown("### Model Operations")
    
    if st.button("🔄 Retrain Energy Model", use_container_width=True):
        with st.spinner("Training energy model..."):
            e_model = train_energy_model()
            save_model(ENERGY_FILE, e_model)
            st.toast("Energy Model Retrained Successfully!", icon="⚡")
            st.rerun()

    if st.button("🤖 Retrain Robot Model", use_container_width=True):
        with st.spinner("Training robot navigation model..."):
            r_model = train_robot_model()
            save_model(ROBOT_FILE, r_model)
            st.toast("Robot Model Retrained Successfully!", icon="🚀")
            st.rerun()

    st.markdown("---")
    st.caption("Reinforcement Learning Engineering Optimizer v2.0")


st.title("🤖 Reinforcement Learning Engineering Optimizer")
st.markdown(
    "Demonstrating Q-learning optimization for **Machine Energy Consumption** "
    "and **Autonomous Robot Grid Navigation**."
)

with st.spinner("Loading models..."):
    energy_model, robot_model = load_or_create_models()

tab1, tab2, tab3 = st.tabs([
    "⚡ Energy Optimization",
    "🤖 Robot Navigation",
    "ℹ️ Project Details"
])


# ---------------------------------------------------------
# TAB 1: ENERGY OPTIMIZATION
# ---------------------------------------------------------

with tab1:
    st.header("⚡ Energy Consumption Optimization")
    st.caption("Learns optimum machine actions (Reduce, Maintain, Increase) for varying loads.")

    energy_q = np.asarray(energy_model["Q"])
    energy_states = energy_model.get("states", ENERGY_STATES)
    energy_actions = energy_model.get("actions", ENERGY_ACTIONS)

    energy_df = pd.DataFrame(
        energy_q,
        index=energy_states,
        columns=energy_actions
    )
    energy_df.index.name = "Machine Load"

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("🔥 Q-Value Heatmap")
        fig_heat = px.imshow(
            energy_df,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            labels=dict(x="Action Taken", y="Machine Load", color="Q-Value"),
            aspect="auto"
        )
        fig_heat.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            font=dict(size=14)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.subheader("🎯 Optimal Policy Recommendations")
        best_actions = []
        for i, state_name in enumerate(energy_states):
            best_index = int(np.argmax(energy_q[i]))
            best_actions.append({
                "Machine Load": state_name,
                "Best Action": energy_actions[best_index],
                "Max Q-Value": f"{energy_q[i, best_index]:.2f}"
            })

        st.table(pd.DataFrame(best_actions))

        st.subheader("🎮 Interactive Recommendation")
        selected_load = st.selectbox(
            "Select Current Machine Load:",
            energy_states,
            key="selected_energy_load"
        )
        selected_index = list(energy_states).index(selected_load)
        best_index = int(np.argmax(energy_q[selected_index]))

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Action for {selected_load} Load</div>
                <div class="metric-value">{energy_actions[best_index]}</div>
                <div class="metric-sub">Expected Reward Value (Q): {energy_q[selected_index, best_index]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📊 Policy Comparison Across States")
    
    fig_bar = px.bar(
        energy_df.reset_index().melt(id_vars="Machine Load", var_name="Action", value_name="Q-Value"),
        x="Machine Load",
        y="Q-Value",
        color="Action",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: ROBOT NAVIGATION
# ---------------------------------------------------------

with tab2:
    st.header("🤖 Robot Navigation Grid World")
    st.caption("5x5 Grid World pathfinding from Start (0,0) to Goal (4,4).")

    path, goal, grid_size = get_robot_path(robot_model)

    # Top KPI Display
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        '<div class="metric-card"><div class="metric-label">Start Point</div><div class="metric-value">(0, 0)</div></div>',
        unsafe_allow_html=True
    )
    m2.markdown(
        f'<div class="metric-card"><div class="metric-label">Goal Point</div><div class="metric-value">{goal}</div></div>',
        unsafe_allow_html=True
    )
    m3.markdown(
        f'<div class="metric-card"><div class="metric-label">Path Length</div><div class="metric-value">{max(0, len(path) - 1)} steps</div></div>',
        unsafe_allow_html=True
    )
    
    status_color = "#34d399" if path[-1] == goal else "#f87171"
    status_text = "Goal Reached" if path[-1] == goal else "Stuck in Loop"
    m4.markdown(
        f'<div class="metric-card"><div class="metric-label">Status</div><div class="metric-value" style="color:{status_color}">{status_text}</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col_grid, col_sim = st.columns([1.2, 1])

    with col_grid:
        st.subheader("🗺️ Path Visualization Map")

        # Build grid visualization matrix
        grid_matrix = np.zeros((grid_size, grid_size))
        path_set = set(path)
        
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) == (0, 0):
                    grid_matrix[r, c] = 1  # Start
                elif (r, c) == goal:
                    grid_matrix[r, c] = 3  # Goal
                elif (r, c) in path_set:
                    grid_matrix[r, c] = 2  # Path

        # Plotly Grid Figure
        fig_grid = go.Figure()

        # Add Heatmap background
        fig_grid.add_trace(go.Heatmap(
            z=grid_matrix,
            colorscale=[
                [0.0, '#1e293b'],   # Empty
                [0.33, '#0284c7'],  # Start
                [0.66, '#f59e0b'],  # Path
                [1.0, '#10b981']    # Goal
            ],
            showscale=False
        ))

        # Annotate Grid text and Path line
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]

        fig_grid.add_trace(go.Scatter(
            x=path_cols,
            y=path_rows,
            mode='lines+markers',
            marker=dict(size=16, color='#ffffff', symbol='circle'),
            line=dict(color='#f59e0b', width=4),
            name="Learned Path"
        ))

        # Label start & goal
        fig_grid.add_annotation(x=0, y=0, text="START", showarrow=False, font=dict(color="white", weight="bold"))
        fig_grid.add_annotation(x=goal[1], y=goal[0], text="GOAL", showarrow=False, font=dict(color="white", weight="bold"))

        fig_grid.update_layout(
            yaxis=dict(autorange="reversed", tickvals=list(range(grid_size))),
            xaxis=dict(tickvals=list(range(grid_size))),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig_grid, use_container_width=True)

    with col_sim:
        st.subheader("🎬 Step-by-Step Path Simulation")
        
        step = st.slider("Step Tracker", 0, len(path) - 1, len(path) - 1)
        current_pos = path[step]

        st.markdown(
            f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-label">Position at Step {step}</div>
                <div class="metric-value">Row {current_pos[0]}, Col {current_pos[1]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("**Full Trajectory Sequence:**")
        path_str = " ➔ ".join([f"`({r},{c})`" for r, c in path])
        st.markdown(path_str)


# ---------------------------------------------------------
# TAB 3: ABOUT PROJECT
# ---------------------------------------------------------

with tab3:
    st.header("ℹ️ Project Architecture & RL Theory")
    
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📌 Model Highlights")
        st.markdown("""
        * **Framework:** Q-Learning (Model-Free Temporal Difference Reinforcement Learning)
        * **Energy Module:** Optimizes dynamic machine load states to balance work output vs energy consumption.
        * **Robot Navigation Module:** Optimizes spatial route navigation in a discrete 2D grid world.
        * **Local Storage:** Persistence using standard Python `pickle` objects.
        """)

    with col_b:
        st.subheader("📐 Mathematical Formulation")
        st.latex(r"""
        Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
        """)
        st.markdown("""
        * **$\alpha$ (Learning Rate):** $0.1$
        * **$\gamma$ (Discount Factor):** $0.9$
        * **$\epsilon$ (Epsilon-Greedy):** $0.2$
        """)

st.divider()
st.caption("⚡ Reinforcement Learning Engineering Optimizer | Created with Streamlit & Plotly")