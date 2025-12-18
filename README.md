# AI Robot 3D Simulator

<p align="center">
  <img src="images/RobotLogo.png" width="110" alt="AI Robot 3D Simulator Logo" />
</p>

<p align="center">
  <b>Advanced Self‑Learning Industrial Robot Arm Simulation Platform</b><br/>
  <i>Artificial Intelligence • Robotics • Real‑Time 3D Visualization • Interactive Control</i>
</p>

<p align="center">
  <a href="https://github.com/dhakarshailendra829/AI_Robot_3D_Simulator"> Star Repository</a> •
  <a href="#-key-capabilities">Key Capabilities</a> •
  <a href="#-system-overview">System Overview</a> •
  <a href="#-architecture--workflow">Architecture</a> •
  <a href="#-screenshots--visuals">Visuals</a> •
  <a href="#-training--learning-pipeline">Training</a> •
  <a href="#-installation--execution">Installation</a>
</p>

---

### Project Overview

**AI Robot 3D Simulator** is a **professional‑grade robotic arm simulation system** designed to demonstrate how modern **AI models, task‑level planning, and real‑time visualization** can be integrated into a single cohesive platform.

The simulator enables both **manual control** and **autonomous execution** of industrial robotic tasks such as **pick, place, move, and sort**, using a **behavior‑cloning–based machine learning model** trained on structured human–object interaction data.

This project is built as a **portfolio‑ready, industry‑aligned system** that reflects real‑world robotics pipelines — from data ingestion and model training to deployment inside an interactive 3D dashboard.

---

## Key Capabilities

### AI‑Driven Motion Learning

* Behavior cloning using deep neural networks (PyTorch)
* Learns joint trajectories from human and object state features
* Generates smooth, continuous joint‑angle predictions

### Intelligent Task Scheduling

* Supports multiple task primitives: `pick`, `place`, `move`, `sort`
* Task queue with autoplay, pause, reset, and step‑wise execution
* Seamless task transitions for continuous robot operation

### Dual Control Modes

* **Manual Mode**: Direct joint‑level control via UI sliders
* **Autonomous Mode**: Model‑driven execution using trained AI
* Safe interpolation between joint states

### Real‑Time 3D Visualization

* Live robotic arm animation
* Task‑wise trajectory rendering
* Visual feedback synchronized with scheduler and model output

### Live Analytics Dashboard

* Real‑time inference monitoring
* Task execution timelines
* System performance insights

---

## System Overview

The simulator provides a **single interactive environment** where users can:

* Train an AI model for robotic motion
* Schedule and execute industrial tasks
* Observe robot behavior in real time
* Analyze model predictions and system performance

**Core system modules:**

* Task Scheduler
* Learning & Inference Engine
* Robot Environment Simulator
* 3D Visualization Engine
* Analytics & Monitoring Layer

---

## Architecture & Workflow

```
Human + Object State Dataset
            ↓
Feature Engineering & Normalization
            ↓
Behavior Cloning Model (PyTorch)
            ↓
Joint Angle Predictions (joint_0 → joint_5)
            ↓
Robot Environment Simulator
            ↓
Real‑Time 3D Visualization & Streamlit UI
```

This modular architecture allows easy extension toward **reinforcement learning**, **physics‑based simulation**, or **multi‑robot coordination**.

---

## Dataset & Learning Structure

### Input Features

* `timestep`
* `human_0 … human_44` (human pose & interaction signals)
* `obj_0 … obj_4` (object state features)

### Target Outputs

* `joint_0 … joint_5` (robot joint angles)

### Task Metadata

* `task_type` (pick / place / move / sort)

---

## Screenshots & Visuals

<p align="center"><b>Live 3D Robot Arm Visualization</b></p>
<p align="center">
  <img src="images/RobotArm.png" width="760" />
</p>

<p align="center"><b>Task Scheduling & Trajectory Playback</b></p>
<p align="center">
  <img src="images/TaskTrajectory.png" width="760" />
</p>

<p align="center"><b>Analytics & System Monitoring Dashboard</b></p>
<p align="center">
  <img src="images/Analytics.png" width="760" />
</p>

---

## Training & Learning Pipeline

* **Model Type:** Fully Connected Neural Network
* **Learning Method:** Supervised behavior cloning
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Epochs:** 50
* **Batch Size:** 32

The trained model is persisted for inference:

```
trained_models/imitation_model.pt
```

This approach enables the robot to **imitate expert motion trajectories** and generalize to unseen task configurations.

---

## 🛠 Tech Stack (By Role)

### AI / ML Engineering

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib

### Software Engineering

* Modular Python architecture
* Task schedulers & simulation environments
* Joblib / model persistence

### Visualization & UI

* Streamlit
* Plotly (3D visualization & analytics)
* Custom animation & trajectory rendering

### System Utilities

* OS, SYS
* Time
* psutil (resource monitoring)

---

## Installation & Execution

### Clone Repository

```bash
git clone https://github.com/dhakarshailendra829/AI_Robot_3D_Simulator.git
cd AI_Robot_3D_Simulator
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Simulator

```bash
streamlit run app.py
```

---

## Future Enhancements

* Reinforcement Learning–based control policies
* Physics‑aware collision handling
* Multi‑robot coordination
* WebGL / Three.js rendering
* Cloud‑deployed simulation services

---

## 👤 Author

**Shailendra Dhakad**
AI • Robotics • Software Systems Engineering

---
