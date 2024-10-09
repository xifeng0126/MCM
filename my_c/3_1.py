import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 假设 df 是您的 DataFrame
# 定义目标变量，这里我们简单地使用 p1_momt - p2_momt
df = pd.read_csv('Wimbledon_featured_matches_last2.csv')  # 替换为您的文件路径
df['momentum_shift'] = df['p1_momt'] - df['p2_momt']

# 选择特征列
features = ['elapsed_time', 'p1_points_won', 'p2_points_won', 'p1_unf_err', 'p2_unf_err', 'p1_net_pt_won', 'p2_net_pt_won','p1_double_fault','p2_double_fault','p1_ace','p2_ace','p1_winner','p2_winner','rally_count']  # 添加您选择的特征

X = df[features]
y = df['momentum_shift']  # 或者您转换后的二元目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 XGBoost 回归模型
model = xgb.XGBRegressor(objective ='reg:squarederror')
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# 特征重要性
xgb.plot_importance(model)
plt.show()

# # 获取feature importance
# plt.figure(figsize=(15, 5))
# plt.bar(range(len(X)), model.feature_importances_)
# plt.xticks(range(len(X)), X, rotation=-45, fontsize=14)
# plt.title('Feature importance', fontsize=14)
# plt.show()
