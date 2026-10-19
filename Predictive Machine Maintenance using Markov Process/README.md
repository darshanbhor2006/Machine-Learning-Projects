# ⚙️ Predictive Machine Maintenance Using Markov Process

## 📌 Project Overview

**Predictive Machine Maintenance Using Markov Process** is a Python-based project that uses Markov Chain probability modeling to simulate machine health conditions and analyze possible machine failures over time.

In manufacturing industries, unexpected machine breakdowns can lead to production delays, equipment damage, and increased maintenance costs. This project demonstrates how a Markov Process can model machine degradation and support preventive maintenance planning.

The project includes an interactive **Streamlit Dashboard** that allows users to select the initial machine health state, adjust simulation time steps, view machine health transitions, and analyze the transition probability matrix.

## 🎯 Objectives

- To understand the application of Markov Processes in predictive maintenance.
- To represent machine health using different operating states.
- To define transition probabilities between machine health conditions.
- To simulate machine health transitions over time.
- To visualize machine degradation using graphs.
- To analyze whether a machine reaches the failure state during a simulation.
- To demonstrate how probability-based modeling can support maintenance planning.

## ✨ Key Features

- **Interactive Dashboard:** A user-friendly dashboard developed using Streamlit.
- **Machine Health States:** Represents Operational, Minor Wear, Major Fault, and Failure.
- **Simulation Controls:** Allows users to select the initial machine state and simulation duration.
- **Markov Chain Simulation:** Simulates machine health transitions using a transition probability matrix.
- **State Transition Log:** Displays the machine health state at each simulation step.
- **Machine Degradation Graph:** Visualizes changes in machine health over time.
- **Transition Probability Matrix:** Displays the probabilities of moving between machine health states.
- **Performance Metrics:** Shows the initial state, final state, number of simulation steps, and whether failure occurred during the simulated history.
- **Industrial-Themed Interface:** Includes a customized background, styled metric cards, tables, and dashboard tabs.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| NumPy | Probability matrix and random state simulation |
| Pandas | Organizing and displaying simulation data |
| Matplotlib | Visualizing machine health transitions |
| Jupyter Notebook | Developing and exploring the project |

## 🧠 What Is a Markov Process?

A Markov Process is a mathematical model used to describe how a system changes from one state to another over time.

It follows the **Markov Property**, which assumes that the next state depends on the current state rather than the complete sequence of previous states.

In this project, the machine can move between four health states:

1. **Operational:** The machine is functioning normally.
2. **Minor Wear:** The machine shows early signs of deterioration.
3. **Major Fault:** The machine has a serious fault and may require maintenance.
4. **Failure:** The machine has reached the failure state.

The model uses a transition probability matrix to determine the probability of moving from one state to another.

## 🔢 Transition Probability Matrix

The project uses the following transition probability matrix:

| Current State | Operational | Minor Wear | Major Fault | Failure |
|---|---:|---:|---:|---:|
| Operational | 0.7 | 0.2 | 0.1 | 0.0 |
| Minor Wear | 0.2 | 0.5 | 0.2 | 0.1 |
| Major Fault | 0.0 | 0.3 | 0.4 | 0.3 |
| Failure | 0.0 | 0.0 | 0.0 | 1.0 |

Each row represents the current machine state, and each column represents the possible next state.

The Failure state is absorbing in this model, meaning that once the machine reaches Failure, it remains there.

## 🔄 Project Workflow

1. Define the machine health states.
2. Create the transition probability matrix.
3. Select the initial machine health state.
4. Choose the number of simulation time steps.
5. Simulate the machine's health transitions using NumPy.
6. Record each state in a transition log.
7. Display the initial state, final state, and failure indicator.
8. Visualize machine health changes using Matplotlib.
9. Display the transition matrix and machine health state information in the dashboard.
10. Analyze the simulation results to understand potential machine degradation.

## 📂 Project Structure

```text
Predictive-Machine-Maintenance/
│
├── app.py
├── Predictive Machine Maintenance using Markov Process.ipynb
└── README.md
```

### File Description

- **`app.py`** — Contains the Streamlit application, dashboard interface, simulation controls, Markov transition probability matrix, machine health simulation, state transition log, and visualizations.
- **`Predictive Machine Maintenance using Markov Process.ipynb`** — Jupyter Notebook for exploring the project, probability modeling, and simulation.
- **`README.md`** — Provides project documentation, features, installation instructions, workflow, and usage details.

## 💻 Requirements

- Python 3.10 or a compatible Python version
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook (optional, for running the notebook)

## 🚀 Installation and Execution

### Step 1: Clone the GitHub Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone https://github.com/darshanbhor2006/Machine-Learning-Projects
```

### Step 2: Open the Project Folder

```bash
cd Predictive-Machine-Maintenance
```

### Step 3: Install the Required Libraries

```bash
pip install streamlit numpy pandas matplotlib
```

To run the Jupyter Notebook as well, install Jupyter:

```bash
pip install jupyter
```

### Step 4: Run the Streamlit Application

Execute the following command in the terminal from the project directory:

```bash
streamlit run app.py
```

The dashboard should open in your default browser. If it does not, open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### Step 5: Use the Dashboard

1. Select the initial machine state from the sidebar.
2. Adjust the simulation time steps using the slider.
3. View the machine health metrics.
4. Explore the state transition log.
5. Analyze the machine degradation graph.
6. Open the second tab to view the health states, transition probability matrix, and conclusion.

### Step 6: Run the Jupyter Notebook (Optional)

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `Predictive Machine Maintenance using Markov Process.ipynb` and run its cells in order.

## 📊 Dashboard Sections

### 1. Simulation and Visualization

This section displays the state transition log and a line graph showing the machine health state at each simulation step.

### 2. Matrix, Setup and Conclusion

This section displays the four machine health states, the transition probability matrix, and the project's conclusion.

### 3. Summary Metrics

The dashboard displays:

- Initial Machine State
- Final Machine State
- Total Simulation Steps
- Failure Triggered: Yes or No

## 🏭 Applications

- Manufacturing equipment maintenance
- Industrial machine reliability analysis
- Preventive maintenance planning
- Machine degradation simulation
- Failure risk modeling
- Smart manufacturing and Industrial IoT research

## ⚠️ Limitations

- The transition probabilities are predefined rather than learned from real machine sensor data.
- The simulation uses random state transitions, so results can vary between runs.
- The dashboard is a simulation and does not directly monitor physical machinery.
- The failure indicator reports whether the Failure state appears in the recorded simulation history; it is not a calibrated real-world failure prediction.
- Real-world deployment would require historical machine data, validation, and appropriate maintenance rules.

## 🔮 Future Scope

- Integrate real-time industrial sensor data.
- Estimate transition probabilities from historical maintenance records.
- Add machine temperature, vibration, and pressure monitoring.
- Generate alerts when a machine enters a high-risk condition.
- Develop maintenance recommendations based on machine health.
- Compare Markov Chain simulation with other predictive maintenance techniques.
- Deploy the dashboard to a cloud platform for remote access.

## 🎓 Learning Outcomes

Through this project, I explored:

- Markov Chains and probability theory.
- Machine health state-transition modeling.
- Python programming and random simulation.
- Data handling using Pandas and NumPy.
- Data visualization using Matplotlib.
- Interactive dashboard development using Streamlit.
- Applications of probability modeling in industrial maintenance.

## 👨‍💻 Author

**Darshan Bhor**

GitHub: https://github.com/darshanbhor2006

## 📜 License

This project is developed for educational and learning purposes. An open-source license can be added to the repository if permission for reuse and modification is intended.
