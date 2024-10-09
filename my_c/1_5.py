import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv("Wimbledon_featured_matches.csv")

# 数据预处理
score_mapping = {'0': 0, '15': 1, '30': 2, '40': 3, '50': 4}
data['p1_score'] = data['p1_score'].map(score_mapping)
data['p2_score'] = data['p2_score'].map(score_mapping)

# 选择特征
features = ['server', 'p1_score', 'p2_score', 'p1_points_won', 'p2_points_won']

X = data[features].values
y = data['point_victor'].values

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 定义神经网络模型
class TennisModel(nn.Module):
    def __init__(self, input_dim):
        super(TennisModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# 创建模型实例
input_dim = X_train.shape[1]
model = TennisModel(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test.unsqueeze(1))
        test_losses.append(test_loss.item())

# 可视化训练和测试损失
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(test_losses, label='Test Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')
plt.show()

# 预测比赛流程中的获胜者
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# 生成可视化图表
plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predicted Winner', marker='o')
#plt.plot(y_test.numpy(), label='Actual Winner', marker='x')
plt.xlabel('Time Point')
plt.ylabel('Winner (1: Player 1, 2: Player 2)')
plt.legend()
plt.title('Match Flow Visualization')
plt.show()
