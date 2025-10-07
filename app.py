# app.py
import sys, os
import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scheduler import generate_tasks
from robot_env import RobotEnv
from visualization import (
    plot_2d_robot,
    plot_3d_robot,
    plot_performance_graph,
    plot_end_effector_path,
    plot_performance_bar_3d
)

st.set_page_config(page_title="Advanced Smart Robot Simulator", layout="wide")

st.title("🤖 Advanced Self-Learning Industrial Robot Simulator")
st.markdown("""
This simulator demonstrates an **AI-powered robotic arm** that learns and executes tasks.
It includes:
- Intelligent task scheduling  
- 2D & 3D trajectory visualization  
- Manual joint control   
- 3D path & performance analysis  
""")

input_dim = 50
output_dim = 6
device = torch.device("cpu")

class ImitationModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

model = ImitationModel(input_dim, output_dim)
model_path = os.path.join("trained_models", "imitation_model.pt")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    st.success("Trained imitation model loaded")
else:
    st.warning("Model file not found. Using untrained model.")

model.eval()

st.sidebar.header("⚙️ Task Scheduler Settings")
num_tasks = st.sidebar.number_input("Number of Tasks", min_value=1, max_value=20, value=5)
task_types = st.sidebar.multiselect(
    "Select Task Types", ["pick", "place", "move", "sort"],
    default=["pick", "place", "move", "sort"]
)

st.sidebar.header("Manual Joint Control")
joint_sliders = [st.sidebar.slider(f"Joint {i}", -np.pi, np.pi, 0.0) for i in range(output_dim)]

robot = RobotEnv(num_joints=output_dim)
robot.reset()

if st.sidebar.button("Generate & Run Simulation"):
    tasks_df = generate_tasks(num_tasks, task_types)
    st.subheader("📋 Scheduled Tasks")
    st.dataframe(tasks_df)

    robot_actions = []
    end_effector_positions = []
    task_colors = []

    # Color mapping for tasks
    task_color_map = {
        "pick": "green",
        "place": "blue",
        "move": "orange",
        "sort": "purple"
    }

    # Predict Actions per Task
    for idx, row in tasks_df.iterrows():
        features = row[['human_' + str(i) for i in range(45)] + ['obj_' + str(i) for i in range(5)]].values.astype(np.float32)
        action = model(torch.from_numpy(features)).detach().numpy()
        robot_actions.append(action)
        robot.step(action)

        # End-effector position calculation
        x, y, z = 0, 0, 0
        angle = 0
        for i, a in enumerate(action):
            angle += a
            x += np.cos(angle)
            y += np.sin(angle)
            z = i * 0.5
        end_effector_positions.append((x, y, z))

        ttype = row['task_type'] if 'task_type' in row else 'move'
        task_colors.append(task_color_map.get(ttype, "gray"))

    st.write("Predicted Robot Joint Angles")
    st.dataframe(pd.DataFrame(robot_actions, columns=[f'joint_{i}' for i in range(output_dim)]))

    # ----------------------------- 2D Visualization -----------------------------
    plot_2d_robot(robot_actions[-1], num_joints=output_dim, color="purple", title="📐 2D Robot Arm Visualization")

    # ----------------------------- 3D Visualization -----------------------------
    plot_3d_robot(
        robot_actions,
        num_joints=output_dim,
        title="🖥️ 3D Robot Arm Animation (Play / Stop / Reset)",
        end_effector_positions=end_effector_positions,
        task_colors=task_colors
    )

    # ----------------------------- 3D End-Effector Path -----------------------------
    plot_end_effector_path(end_effector_positions, task_colors)

    # ----------------------------- Performance Graph -----------------------------
    plot_performance_graph(robot_actions, title="📊 Robot Action Magnitude per Task", color="magenta")

    # ----------------------------- 3D Performance (Cylinder/Box workaround) -----------------------------
    plot_performance_bar_3d(robot_actions, tasks_df)

# --------------------------------------------------------------------
# 🕹️ Manual Joint Control
# --------------------------------------------------------------------
st.subheader("🎛️ Manual Joint Control")
manual_action = np.array(joint_sliders)
plot_2d_robot(manual_action, num_joints=output_dim, color="orange", title="🎛️ Manual Joint Control")
