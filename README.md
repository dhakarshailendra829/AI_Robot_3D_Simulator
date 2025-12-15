# AI Robot 3D Simulator
<p align="center">
  <img src="images/robot_banner.png" width="800" alt="AI Robot 3D Simulator" />
</p>

<p align="center">
  <b>Advanced Self‑Learning Industrial Robot Arm Simulator</b><br/>
  <i>AI‑powered training • Real‑time 3D visualization • Interactive control dashboard</i>
</p>

<p align="center">
  <a href="https://github.com/dhakarshailendra829/AI_Robot_3D_Simulator">Star Repo</a> •
  <a href="#-features">Features</a> •
  <a href="#-screenshots">Screenshots</a> •
  <a href="#-system-overview">Overview</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-training--model">Model</a> •
  <a href="#-ui--visualization">UI</a>
</p>

---

## Project Summary
**AI Robot 3D Simulator** is a professional‑grade, self‑learning robotic arm simulation platform that demonstrates **intelligent robotic control, task scheduling, and real‑time 3D visualization** in a single unified system.
The project combines **behavior‑cloning–based machine learning**, **interactive dashboards**, and **live 3D animations** to simulate industrial robot tasks such as **pick, place, move, and sort**. It supports both **manual joint‑level control** and **fully autonomous execution** using a trained AI model.
This repository is designed as a **portfolio‑ready, research‑oriented showcase** of AI + Robotics + Visualization engineering.
---

## Features
### AI‑Powered Learning
* Behavior cloning using deep neural networks (PyTorch)
* Learns robot joint trajectories from human & object state data
* Predicts smooth joint movements in real time

### Intelligent Task Scheduling
* Multiple task types: `pick`, `place`, `move`, `sort`
* Task queue with autoplay, pause, and step execution
* Seamless transition between consecutive tasks

### Robot Control Modes
* Manual joint control via UI sliders
* Autonomous execution using trained model
* Safe, smooth joint interpolation

### Real‑Time 3D Visualization
* Live robot arm animation
* 3D trajectory rendering per task
* Visual feedback synced with task execution

### Live Analytics Dashboard
* Real‑time performance metrics
* System resource monitoring
* Training & inference insights
---

## 🖼 Screenshots
<p align="center"><b>Live 3D Robot Arm Visualization</b></p>
<p align="center">
  <img src="images/RobotArm.png" width="700" />
</p>

<p align="center"><b>Task Scheduling & Autoplay Control</b></p>
<p align="center">
  <img src="images/TaskTrajectory.png" width="700" />
</p>

<p align="center"><b>Analytics & Performance Dashboard</b></p>
<p align="center">
  <img src="images/Analytics.png" width="700" />
</p>
> 
---

## System Overview
The simulator provides a **single interactive interface** where users can:
* Train a learning model
* Schedule robot tasks
* Visualize motion in real time
* Analyze system and model performance

Core modules:
* Scheduling
* Training
* 3D View
* Analytics
* Manual Control
---

## Architecture
```
Human + Object State Dataset
        ↓
Behavior Cloning Model (PyTorch)
        ↓
Joint Angle Predictions (joint_0 → joint_5)
        ↓
Robot Environment Simulator
        ↓
3D Visualization + Streamlit UI Dashboard
```

---
## Dataset Structure
**Input Features**

* `timestep`
* `human_0 ... human_44`
* `obj_0 ... obj_4`

**Target Outputs**

* `joint_0 ... joint_5`

**Task Metadata**

* `task_type`
---

## Tech Stack
### Backend & AI
* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib

### UI & Visualization
* Streamlit
* Plotly (3D graphs & analytics)
* Custom visualization engine

### System Utilities
* OS, SYS
* Time
* psutil (performance monitoring)
---

## Installation

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

## Training & Model
* Model type: Fully connected neural network
* Loss: Mean Squared Error (MSE)
* Optimizer: Adam
* Epochs: 50
* Batch size: 32
The trained model is stored at:
```
trained_models/imitation_model.pt
```
---

## UI & Visualization
The Streamlit UI enables:
* Live 3D robot visualization
* Autoplay / pause task execution
* Manual joint manipulation
* Real‑time analytics plots
* Smooth animation transitions
---

## Future Enhancements
* Reinforcement Learning (RL) based control
* Physics‑based collision handling
* Multi‑robot coordination
* WebGL / Three.js rendering
* Cloud‑based deployment
---

## Author
**Shailendra Dhakad**
AI • Robotics • Systems Engineering
---
