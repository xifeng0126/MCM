import torch
import torch.nn as nn

# 定义一个简单的GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # 前向传播函数
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out, h


# 创建一个GRU模型实例
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1
model = GRUModel(input_size, hidden_size, num_layers, output_size)

# 定义输入数据和初始隐藏状态
input_data = torch.randn(5, 3, 10)  # 输入数据形状为 (batch_size, seq_len, input_size)
hidden = torch.zeros(num_layers, 5, hidden_size)  # 初始隐藏状态形状为 (num_layers, batch_size, hidden_size)

# 调用模型进行前向传播
output, hidden = model(input_data, hidden)
print(output)
