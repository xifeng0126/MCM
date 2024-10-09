import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 加载数据，假设文件名为'your_data_file.xlsx'
df = pd.read_excel('data.xlsx')

# 确保日期格式正确，并将其设置为索引
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_values(by='Date')

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
df['Num'] = scaler.fit_transform(df['Num'].values.reshape(-1, 1))

# 创建序列数据集
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data.iloc[i:(i+seq_length)].values
        y = data.iloc[i+seq_length].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 30  # 使用30天的数据预测下一天，滑动窗口
X, y = create_sequences(df[['Num']], seq_length)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 数据集分割
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_layer_size)
        out, _ = self.gru(x, h0)
        out = self.linear(out[:, -1, :])
        return out


model = GRUModel(input_size=1, hidden_layer_size=100, output_size=1)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_function(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss {loss.item()}')

model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = loss_function(preds, y_test)
print(f'Test Loss: {test_loss.item()}')
# 将预测值转换回原始数据范围（如果之前进行了归一化或标准化）
predictions = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

# 将实际值也转换回原始数据范围
actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
devitation = np.abs(predictions - actual)
average_raletiave_error = np.mean(devitation / actual * 100)

print(f'Average Relative Error: {average_raletiave_error:.2f}%')

# 绘制预测值和实际值
plt.figure(figsize=(10, 6))
plt.plot(actual, label='Actual', color='blue', marker='o')
plt.plot(predictions, label='Predicted', color='red')
plt.plot(devitation, label='Deviation', color='green', marker='.')
plt.fill_between(range(len(devitation)), devitation, color='green', alpha=0.1)
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()

def predict_future(date, model, last_seq, scaler):
    model.eval()
    with torch.no_grad():
        future_seq = last_seq.unsqueeze(0)  # 添加批处理维度
        predicted_num = model(future_seq).item()
        predicted_num = scaler.inverse_transform([[predicted_num]])[0][0]
    return predicted_num

# 选择最后seq_length天作为输入
last_seq = X[-1]
future_date = pd.to_datetime('2023-03-01')
predicted_num = predict_future(future_date, model, last_seq, scaler)
print(f'Predicted number of answers on {future_date.strftime("%Y-%m-%d")}: {int(predicted_num)}')
