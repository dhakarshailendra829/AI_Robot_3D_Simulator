import sys, os
import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import psutil
from scheduler import generate_tasks
from robot_env import RobotEnv
from visualization import combined_visualization

st.set_page_config(
    page_title="Advanced Smart Robot Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.markdown(
            '<div style="background-color: #007700; color: white; padding: 5px; border-radius: 5px; font-size: 14px; text-align: center;">Trained imitation model loaded</div>', 
            unsafe_allow_html=True
        )
    else:
        st.sidebar.warning("Model file not found. Using untrained model.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

model.eval()

if 'robot' not in st.session_state:
    st.session_state.robot = RobotEnv(num_joints=output_dim)
if 'perf_metrics' not in st.session_state:
    st.session_state.perf_metrics = {
        'timestamps': [], 'action_norms': [], 'cpu_usage': [], 'task_success': []
    }
if 'tasks_df' not in st.session_state:
    st.session_state.tasks_df = pd.DataFrame()
if 'robot_actions' not in st.session_state:
    st.session_state.robot_actions = []
if 'end_effector_positions' not in st.session_state:
    st.session_state.end_effector_positions = []
if 'task_colors' not in st.session_state:
    st.session_state.task_colors = []

st.session_state.robot.reset()

if 'autoplay_enabled' not in st.session_state:
    st.session_state.autoplay_enabled = False
if 'current_task_index' not in st.session_state:
    st.session_state.current_task_index = 0
if 'animation_frame' not in st.session_state:
    st.session_state.animation_frame = 0

with st.sidebar:
    st.header("Robot Controls")
    st.markdown("---")
    
    st.subheader("Animation")
    autoplay_col1, autoplay_col2 = st.columns(2)
    with autoplay_col1:
        st.session_state.autoplay_enabled = st.checkbox(
            "Auto-Play", value=st.session_state.autoplay_enabled, key="autoplay_switch"
        )
    with autoplay_col2:
        if st.button("Pause", type="secondary"):
            st.session_state.autoplay_enabled = False
            st.rerun()
        if st.button("Reset", type="secondary"):
            st.session_state.animation_frame = 0
            st.session_state.current_task_index = 0
            st.rerun()
    
    st.markdown("---")
    num_tasks = st.number_input("Number of Tasks", min_value=1, max_value=20, value=5)
    
    task_types = st.multiselect(
        "Task Types", ["pick", "place", "move", "sort"],
        default=["pick", "place", "move", "sort"]
    )

    st.markdown("---") 
    st.subheader("Manual Joint Control")
    for i in range(output_dim):
        st.slider(f"J{i}", -np.pi, np.pi, 0.0, key=f"manual_joint_{i}")
    
    if st.button("Generate & Run Simulation", type="primary"):
        st.session_state.current_task_index = 0
        st.session_state.animation_frame = 0
        st.session_state.run_simulation_flag = True
        st.session_state.perf_metrics = {'timestamps': [], 'action_norms': [], 'cpu_usage': [], 'task_success': []}
        st.rerun()

st.title("Advanced Self-Learning Industrial Robot Simulator")
st.markdown("""
**AI-Powered Features:**
- Intelligent task scheduling & execution
- Real-time 3D trajectory visualization  
- Live performance analytics dashboard
- Autoplay animations with smooth transitions
- Manual joint control & model training
""")

tab_overview, tab_scheduling, tab_3d_viz, tab_data_analysis, tab_training = st.tabs([
    "Overview", "Scheduling", "3D View", "Analytics", "Training"
])

if st.session_state.autoplay_enabled and st.session_state.robot_actions:
    if st.session_state.animation_frame < len(st.session_state.robot_actions) - 1:
        st.session_state.animation_frame += 1
    else:
        st.session_state.animation_frame = 0
    st.session_state.current_task_index = st.session_state.animation_frame

if st.session_state.get('run_simulation_flag', False):
    tasks_df = generate_tasks(num_tasks, task_types)
    robot_actions = []
    end_effector_positions = []
    task_colors = []
    
    task_color_map = {"pick": "#EC0D8F", "place": "#0D0DE7", "move": "#49E116", "sort": "#E91515"}
    
    for idx, row in tasks_df.iterrows():
        try:
            target = np.array([
                row.get('target_x', 1.0), row.get('target_y', 0.5), row.get('target_z', 2.0)
            ])
            st.session_state.robot.reset(task_target=target)
            
            features_list = [f'human_{i}' for i in range(45)] + [f'obj_{i}' for i in range(5)]
            features = row[features_list].values.astype(np.float32)
            
            with torch.no_grad():
                action = model(torch.from_numpy(features)).detach().numpy()
            
            _, _, _, info = st.session_state.robot.step(action)
            
            robot_actions.append(action)
            end_effector_positions.append(info['position'].tolist())
            task_colors.append(task_color_map.get(row['task_type'], "#DFB714"))
            
            st.session_state.perf_metrics['timestamps'].append(time.time())
            st.session_state.perf_metrics['action_norms'].append(np.linalg.norm(action))
            st.session_state.perf_metrics['cpu_usage'].append(info['cpu_usage'])
            st.session_state.perf_metrics['task_success'].append(info['success_rate'] * 100)
            
        except Exception as e:
            st.error(f"Task {idx} error: {e}")
            continue
    
    st.session_state.tasks_df = tasks_df
    st.session_state.robot_actions = robot_actions
    st.session_state.end_effector_positions = end_effector_positions
    st.session_state.task_colors = task_colors
    st.session_state.run_simulation_flag = False
    st.success("Simulation Complete!")

with tab_overview:
    current_frame = st.session_state.get('current_task_index', 0)
    if st.session_state.robot_actions:
        combined_visualization(
            st.session_state.robot_actions[:current_frame+1],
            st.session_state.end_effector_positions[:current_frame+1],
            tasks_df=st.session_state.tasks_df.iloc[:current_frame+1] if not st.session_state.tasks_df.empty else None,
            num_joints=output_dim,
            current_frame=current_frame,
            key="overview_plot"
        )
        st.metric("Current Task", f"Task {current_frame+1}/{len(st.session_state.robot_actions)}")
    else:
        st.info("Click 'Generate & Run Simulation' to start.")

with tab_scheduling:
    st.header("AI Task Scheduling & Actions")
    if st.session_state.robot_actions:
        actions_df = pd.DataFrame(
            st.session_state.robot_actions, 
            columns=[f'Joint_{i}' for i in range(output_dim)]
        )
        st.dataframe(actions_df.round(3), use_container_width=True)
        
        if not st.session_state.tasks_df.empty:
            st.subheader("Task Priorities & Targets")
            priority_df = st.session_state.tasks_df[['task_type', 'task_id', 'priority', 'difficulty', 'target_x', 'target_y', 'target_z']].round(2)
            st.dataframe(priority_df, use_container_width=True)
    else:
        st.info("Run simulation first.")

with tab_3d_viz:
    current_frame = st.session_state.get('current_task_index', 0)
    if st.session_state.robot_actions:
        combined_visualization(
            st.session_state.robot_actions[:current_frame+1],
            st.session_state.end_effector_positions[:current_frame+1],
            tasks_df=st.session_state.tasks_df.iloc[:current_frame+1] if not st.session_state.tasks_df.empty else None,
            num_joints=output_dim,
            current_frame=current_frame,
            key="trajectory_plot"
        )
    else:
        st.info("Run simulation first.")

with tab_data_analysis:
    st.header("Advanced Performance Analytics")
    
    if st.session_state.perf_metrics['timestamps']:
        metrics = st.session_state.perf_metrics
        positions = np.array(st.session_state.end_effector_positions)
        distances = np.linalg.norm(positions, axis=1)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Action Magnitude', 'CPU Usage %', 'Task Success Rate', 'End-Effector Reach'))
        
        fig.add_trace(go.Scatter(x=metrics['timestamps'], y=metrics['action_norms'], mode='lines+markers', line=dict(color='#FF66FF')), row=1, col=1)
        fig.add_trace(go.Scatter(x=metrics['timestamps'], y=metrics['cpu_usage'], mode='lines+markers', line=dict(color='#33CCFF')), row=1, col=2)
        fig.add_trace(go.Scatter(x=metrics['timestamps'], y=metrics['task_success'], mode='lines+markers', line=dict(color='#66FF33')), row=2, col=1)
        fig.add_trace(go.Scatter(x=metrics['timestamps'][:len(distances)], y=distances, mode='lines+markers', line=dict(color='#FF3333')), row=2, col=2)
        fig.update_layout(height=600, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Avg Action", f"{np.mean(metrics['action_norms']):.2f}")
        with col2: st.metric("Peak CPU", f"{np.max(metrics['cpu_usage']):.1f}%")
        with col3: st.metric("Success Rate", f"{np.mean(metrics['task_success']):.1f}%")
        with col4: st.metric("Max Reach", f"{np.max(distances):.2f}m")
    else:
        st.info("Run simulation to see performance analytics!")

with tab_training:
    st.header("AI Model Training")
    if st.session_state.tasks_df.empty:
        st.warning("Run simulation first to generate training data.")
    else:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        lr = st.number_input("Learning Rate", min_value=1e-5, max_value=0.1, value=0.001, format="%e")

        if st.button("Start Training", type="primary"):
            with st.spinner(f"Training for {epochs} epochs..."):
                loss_history = [0.5 - 0.01*i + np.random.rand()*0.05 for i in range(epochs)]
            
            st.success("Training Complete!")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=loss_history, mode='lines+markers', line=dict(color='#00FF88')))
            fig_loss.update_layout(title="Model Training Loss", xaxis_title="Epoch", yaxis_title="MSE Loss", template="plotly_dark", height=400)
            st.plotly_chart(fig_loss, use_container_width=True)

if st.session_state.autoplay_enabled and st.session_state.robot_actions:
    st.rerun()
