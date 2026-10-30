# 🤖 Reinforcement Learning for Optimizing Engineering Designs

## 📌 Project Overview

This project demonstrates the application of **Reinforcement Learning (RL)** to optimize engineering processes using the Q-Learning algorithm.

The system uses an intelligent agent that learns through interaction with its environment and selects the best possible actions based on rewards and penalties.

The project focuses on two applications:

1. **Energy Consumption Optimization:** Optimizes machine operations based on different energy consumption or load conditions.
2. **Robot Navigation:** Helps a robot learn the shortest possible path from a starting position to a target position in a grid environment.

A Streamlit web application is developed to visualize the results, display Q-tables, show optimal policies, and demonstrate robot navigation.

## 🎯 Project Objectives

- Understand the concept of Reinforcement Learning.
- Implement the Q-Learning algorithm.
- Optimize energy consumption through intelligent decision-making.
- Develop an intelligent robot navigation system.
- Store trained Q-Learning models using Pickle files.
- Develop an interactive web application using Streamlit.

## ⚙️ Technologies Used

- Python
- Reinforcement Learning
- Q-Learning Algorithm
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Pickle
- Jupyter Notebook
- VS Code

## 🧠 Applications

### 1. Energy Consumption Optimization

This module uses Q-Learning to select suitable machine operating actions based on the current load condition.

**Machine Load Conditions:**

- Low
- Medium
- High

**Available Actions:**

- Reduce
- Maintain
- Increase

The agent learns the best action for each machine load condition using rewards and Q-values.

### 2. Robot Navigation (Grid World)

This module demonstrates how a robot learns to navigate through a 5 × 5 grid environment.

**Features:**

- Starting Position: (0, 0)
- Goal Position: (4, 4)
- Movement: Up, Down, Left, Right
- Reward-based learning
- Q-table-based action selection
- Learned path visualization

The robot learns to reach the target position by maximizing rewards.

## 📐 Q-Learning Algorithm

Q-Learning is a model-free Reinforcement Learning algorithm that learns the best action for each state.

**Q-Learning Update Formula:**

Q(s, a) = Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]

Where:

- Q(s, a) = Current Q-value
- α = Learning rate
- r = Reward received
- γ = Discount factor
- s = Current state
- a = Selected action
- s' = Next state
- a' = Next action

## 📂 Project Structure

```text
Reinforcement_Learning_Engineering/
│
├── Implementation.ipynb
│
├── app.py
│
├── energy_optimization.pkl
│
├── robot_navigation.pkl
│
└── README.md
```

## 🚀 Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/darshanbhor2006
```

### Step 2: Navigate to the Project Folder

```bash
cd Reinforcement_Learning_Engineering
```

### Step 3: Install Required Libraries

```bash
pip install streamlit numpy pandas matplotlib
```

### Step 4: Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `Implementation.ipynb` and run the cells to execute the Reinforcement Learning implementations.

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal:

```text
http://localhost:8501
```

## 💾 Model Saving Using Pickle

The project uses Python's Pickle module to save trained Q-Learning models.

**Saved Models:**

- `energy_optimization.pkl` – Stores the energy optimization Q-table and related information.
- `robot_navigation.pkl` – Stores the robot navigation Q-table and related information.

These files allow the application to load previously trained models without repeating the training process.

If the model files are unavailable, the Streamlit application automatically trains and saves new models.

## 📊 Streamlit Application Features

- Interactive dashboard.
- Energy optimization Q-table.
- Optimal machine operating recommendations.
- Robot navigation grid.
- Learned robot path.
- Model loading and saving.
- Robot model retraining option.
- Project information section.

## 🔮 Future Scope

- Apply Deep Q-Networks (DQN) for complex engineering problems.
- Integrate real-time IoT sensors for energy monitoring.
- Optimize industrial machine performance.
- Introduce obstacles and dynamic environments in robot navigation.
- Develop intelligent industrial automation systems.
- Apply Reinforcement Learning to smart manufacturing and robotics.

## 🎓 Conclusion

This project demonstrates how Reinforcement Learning can be applied to engineering optimization problems. Using the Q-Learning algorithm, the system learns suitable decisions for energy consumption optimization and robot navigation.

The integration of Python, Pickle, and Streamlit provides an interactive platform to train, store, load, and visualize the learned results.

## 👨‍💻 Project Information

**Project Title:** Reinforcement Learning for Optimizing Engineering Designs

**Domain:** Artificial Intelligence and Machine Learning

**Algorithm:** Q-Learning

**Programming Language:** Python

**Application Framework:** Streamlit
