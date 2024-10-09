import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设df是包含比赛数据的DataFrame
df = pd.read_csv('Wimbledon_featured_matches.csv')
# 数据预处理（示例）
df['server_win'] = np.where(df['server'] == df['point_victor'], 1, 0)  # 发球方是否赢得该分
df['is_server'] = np.where(df['server'] == 1, 1, 0)  # 当前球员是否为发球方

# 特征和标签
features = ['is_server', 'serve_no', 'p1_points_won', 'p2_points_won', 'p1_ace', 'p2_ace', 'p1_double_fault', 'p2_double_fault']
X = df[features]
y = df['server_win']  # 使用发球方是否赢得该分作为标签

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用随机森林作为分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# 预测和评估
y_pred = clf.predict(X_test_scaled)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# 势头变化的可视化
# 计算每个点的累积得分差
df['score_diff'] = df['p1_points_won'] - df['p2_points_won']
df['cumulative_diff'] = df['score_diff'].cumsum()

# 可视化比赛流程和势头变化
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['cumulative_diff'], label='Cumulative Score Difference')
plt.axhline(0, color='grey', lw=0.5, linestyle='--')
plt.xlabel('Point Number')
plt.ylabel('Cumulative Score Difference (P1 - P2)')
plt.title('Match Flow and Momentum')
plt.legend()
plt.show()
