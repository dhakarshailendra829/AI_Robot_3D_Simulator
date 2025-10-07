import numpy as np
class RobotEnv:
    def __init__(self, num_joints=6):
        self.num_joints = num_joints
        self.joint_angles = np.zeros(num_joints)
        self.trajectory = []

    def reset(self):
        self.joint_angles = np.zeros(self.num_joints)
        self.trajectory = []
        return self.joint_angles

    def step(self, action):
        self.joint_angles = np.array(action)
        self.trajectory.append(self.joint_angles)
        return self.joint_angles

    def get_end_effector_position(self):
        x, y = [0], [0]
        angle = 0
        for a in self.joint_angles:
            angle += a
            x.append(x[-1] + np.cos(angle))
            y.append(y[-1] + np.sin(angle))
        return x, y
