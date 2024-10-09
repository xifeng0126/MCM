import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('Wimbledon_featured_matches.csv')
# 假设df是包含比赛数据的DataFrame，包含'point_victor', 'server', 'p1_ace', 'p2_ace', 'p1_double_fault', 'p2_double_fault', 'p1_winner', 'p2_winner'等列
df['server'] = df['server'].replace(2, 0)

# 将点数胜利者转换为二元变量
df['p1_point_win'] = df['point_victor'].apply(lambda x: 1 if x == 1 else 0)

# 特征和标签
X = df[['server', 'p1_ace', 'p2_ace', 'p1_double_fault', 'p2_double_fault', 'p1_winner', 'p2_winner']]
y = df['p1_point_win']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 可视化回归系数
plt.figure(figsize=(10, 6))
plt.bar(X.columns, model.coef_[0])
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Regression Coefficients')
plt.xticks(rotation=30)
plt.show()
