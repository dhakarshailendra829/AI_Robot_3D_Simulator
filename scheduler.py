# scheduler.py
import pandas as pd
import numpy as np

def generate_tasks(num_tasks, task_types):
    """
    Generate random tasks with human poses and object positions.
    """
    tasks = []
    for i in range(num_tasks):
        task_type = np.random.choice(task_types)
        human_pose = np.random.rand(45)  # 15 joints * 3
        object_positions = np.random.rand(5)  # 5 objects
        row = list(human_pose) + list(object_positions)
        tasks.append([task_type] + row)
    columns = ['task_type'] + ['human_' + str(i) for i in range(45)] + ['obj_' + str(i) for i in range(5)]
    df = pd.DataFrame(tasks, columns=columns)
    return df
