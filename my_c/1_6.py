import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Wimbledon_featured_matches.csv')

# 创建示例数据（特征矩阵 X 和目标变量 y）
X = df[['p1_ace', 'server', 'p1_winner','p1_net_pt_won']].iloc[:300].values
y = df['p1_perform'].iloc[:300].values

# 创建一个线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 使用模型进行预测
y_pred = model.predict(X)

# 使用Seaborn改善美观
sns.set(style="whitegrid")

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制实际值和预测值的散点图，用不同颜色和标记区分
plt.scatter(y, y_pred, c='deepskyblue', label='Actual vs. Predicted', alpha=0.7, edgecolors='w', s=80)

# 添加标签和标题
plt.xlabel("Actual Value", fontsize=12)
plt.ylabel("Predicted Value", fontsize=12)
plt.title("Multiple Linear Regression Prediction Results", fontsize=14)

# 绘制一条完美拟合线（y=x）
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')

# 添加图例
plt.legend()

plt.savefig('2_300.png', format='png')
# 显示图形
plt.show()
