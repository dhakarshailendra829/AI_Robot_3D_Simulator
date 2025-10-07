import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------------
# 🟣 2D Robot Visualization
# -------------------------------------------------------------------
def plot_2d_robot(action, num_joints, color="purple", title="2D Robot Arm"):
    """
    2D Robot Arm Visualization using Matplotlib
    """
    x, y = [0], [0]
    angle = 0
    for a in action:
        angle += a
        x.append(x[-1] + np.cos(angle))
        y.append(y[-1] + np.sin(angle))
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(x, y, marker='o', linewidth=4, color=color)
    ax.set_xlim(-num_joints-1, num_joints+1)
    ax.set_ylim(-num_joints-1, num_joints+1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(title)
    st.pyplot(fig)

# -------------------------------------------------------------------
# 🟡 3D Robot Visualization with optional extra data
# -------------------------------------------------------------------
def plot_3d_robot(
    robot_actions,
    num_joints,
    colors_map=None,
    title="3D Robot Arm Animation",
    end_effector_positions=None,
    workspace=None,
    obstacles=None,
    velocity_vectors=None,
    task_colors=None
):
    """
    3D Robot Arm Visualization using Plotly with animation frames.
    """
    if colors_map is None:
        colors_map = ["green", "red", "blue", "orange", "purple", "cyan"]

    # Animation frames for each action
    frames = []
    for idx, action in enumerate(robot_actions):
        x3, y3, z3, c3 = [0], [0], [0], []
        angle = 0
        for i, a in enumerate(action):
            angle += a
            x3.append(x3[-1] + np.cos(angle))
            y3.append(y3[-1] + np.sin(angle))
            z3.append(i * 0.5)
            c3.append(colors_map[i % len(colors_map)])
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=x3, y=y3, z=z3,
                mode='lines+markers',
                line=dict(width=10, color='green'),
                marker=dict(size=6, color=c3)
            )],
            name=f'frame{idx}'
        ))

    # Initial pose
    x0, y0, z0, c0 = [0], [0], [0], []
    angle = 0
    for i, a in enumerate(robot_actions[0]):
        angle += a
        x0.append(x0[-1] + np.cos(angle))
        y0.append(y0[-1] + np.sin(angle))
        z0.append(i * 0.5)
        c0.append(colors_map[i % len(colors_map)])

    fig3 = go.Figure(
        data=[go.Scatter3d(
            x=x0, y=y0, z=z0,
            mode='lines+markers',
            line=dict(width=10, color='green'),
            marker=dict(size=6, color=c0)
        )],
        frames=frames
    )

    # Optional: End-effector path
    if end_effector_positions is not None and len(end_effector_positions) > 1:
        xe, ye, ze = zip(*end_effector_positions)
        fig3.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze,
            mode='lines+markers',
            marker=dict(size=4, color='magenta'),
            line=dict(color='gray', width=2),
            name='End Effector Path'
        ))

    # Optional: Task Colors (color-coded joints per task)
    if task_colors is not None and len(task_colors) == len(robot_actions):
        pass  # You can use this to color per task in future if needed

    fig3.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-num_joints-1, num_joints+1]),
            yaxis=dict(range=[-num_joints-1, num_joints+1]),
            zaxis=dict(range=[0, num_joints]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=700, redraw=True),
                                 fromcurrent=True, mode='immediate')]
            )]
        )]
    )

    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------------------------
# 🔸 2D Performance Line Graph
# -------------------------------------------------------------------
def plot_performance_graph(robot_actions, title="Robot Action Magnitude per Task", color="magenta"):
    """
    Plot performance graph: magnitude of joint actions per task
    """
    task_numbers = list(range(1, len(robot_actions)+1))
    magnitudes = [np.linalg.norm(a) for a in robot_actions]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(task_numbers, magnitudes, marker='o', color=color)
    ax.set_xlabel("Task Number")
    ax.set_ylabel("Joint Action Magnitude")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# -------------------------------------------------------------------
# 🟢 End Effector Path Visualization (Optional)
# -------------------------------------------------------------------
def plot_end_effector_path(positions, task_colors):
    x, y, z = zip(*positions)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(size=5, color=task_colors),
        line=dict(color='gray', width=3)
    ))
    fig.update_layout(scene=dict(aspectmode="cube"))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 🟠 3D Performance "Bar" Graph using Scatter3d Lines (No Bar3d)
# -------------------------------------------------------------------
def plot_performance_bar_3d(robot_actions, tasks_df):
    """
    Simulate a 3D Bar Graph using vertical Scatter3d lines
    representing energy (magnitude) for each task.
    """
    energy = [float(np.linalg.norm(a)) for a in robot_actions]
    x = list(range(len(energy)))
    y = [0] * len(energy)
    z0 = [0] * len(energy)

    fig = go.Figure()

    for i, e in enumerate(energy):
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[i]],
            y=[y[i], y[i]],
            z=[0, e],
            mode='lines+markers',
            line=dict(width=10, color='magenta'),
            marker=dict(size=5, color='magenta'),
            name=f"Task {i+1}"
        ))

    fig.update_layout(
        title="📊 3D Task Energy Bar Graph",
        scene=dict(
            xaxis_title='Task Index',
            yaxis_title='(Dummy Axis)',
            zaxis_title='Energy Magnitude'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
