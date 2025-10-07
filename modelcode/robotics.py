import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

df = pd.read_csv('synthetic_robot_dataset.csv')

feature_cols = [col for col in df.columns if col.startswith('human_') or col.startswith('obj_')]
X = df[feature_cols].values.astype(np.float32)

target_cols = [col for col in df.columns if col.startswith('joint_')]
y = df[target_cols].values.astype(np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

class RobotDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = RobotDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class ImitationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImitationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1]
output_dim = y.shape[1]
model = ImitationModel(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

plt.figure(figsize=(8,5))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Behavior Cloning Model Training Loss')
plt.legend()
plt.grid(True)
plt.show()

os.makedirs('trained_models', exist_ok=True)
model_path = 'trained_models/imitation_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Trained model saved at {model_path}")