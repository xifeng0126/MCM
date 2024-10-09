import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 假设 df 是您的 DataFrame
# 定义目标变量，这里我们简单地使用 p1_momt - p2_momt
df = pd.read_csv('Wimbledon_featured_matches_last2.csv')  # 替换为您的文件路径
df['momentum_shift'] = df['p1_momt'] - df['p2_momt']
df['unf_err_diff'] = df['p1_unf_err'] - df['p2_unf_err']
df['net_pt_won_diff'] = df['p1_net_pt_won'] - df['p2_net_pt_won']
df['double_fault_diff'] = df['p1_double_fault'] - df['p2_double_fault']
df['ace_diff'] = df['p1_ace'] - df['p2_ace']
df['winner_diff'] = df['p1_winner'] - df['p2_winner']
# 五个特征
df['points_won_diff'] = df['p1_points_won'] - df['p2_points_won']
df['state_diff'] = df['p1_state'] - df['p2_state']
df['good_point'] = df['ace_diff'] + df['winner_diff'] + df['net_pt_won_diff']
df['bad_point'] = df['double_fault_diff'] + df['unf_err_diff']
df['contin_point'] = ((df['point_victor'] == 1) & (df['last_winner'] == 1)).astype(int) - ((df['point_victor'] == 2) & (df['last_winner'] == 2)).astype(int)


# 选择特征列
features = ['contin_point','points_won_diff', 'state_diff', 'good_point', 'bad_point'] # 添加您选择的特征
X = df[features]
y = df['momentum_shift']  # 或者您转换后的二元目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 XGBoost 回归模型
model = xgb.XGBRegressor(objective ='reg:squarederror')
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)
# 计算MAE
# mae = mean_absolute_error(y_test, y_pred)
# print(f"MAE: {mae}")
#
# 定义扰动函数
# 定义添加噪声的函数
def add_noise_to_dataframe(df, noise_level=0.05):
    noised_df = df.copy()  # 创建DataFrame的副本以避免修改原始数据
    for column in df.columns:
        # 为每列生成噪声
        noise = np.random.randn(len(df[column])) * df[column].std() * noise_level
        noised_df[column] = df[column] + noise
    return noised_df

# 应用函数来添加噪声
X_train_noised = add_noise_to_dataframe(X_test, noise_level=0.05)

# 现在X_train_noised包含了添加了噪声的数据，可以继续用于模型预测和评估


# 使用扰动数据进行预测
y_pred_noised = model.predict(X_train_noised)

# 计算扰动前后的预测结果差异
mae_noised = mean_absolute_error(y_test, y_pred_noised)
print(f"MAE with noised data: {mae_noised}")

# 对比原始MAE
y_pred_original = model.predict(X_test)
mae_original = mean_absolute_error(y_test, y_pred_original)
print(f"Original MAE: {mae_original}")

# 分析结果差异
print(f"Difference in MAE: {abs(mae_noised - mae_original)}")

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置Seaborn样式
sns.set(style="whitegrid")

# 数据准备
mae_values = [0.20657118410042083, 0.2113404537849486]  # 分别是原始MAE和噪声干扰后的MAE
mae_labels = ['Original', 'With Noise']

# 创建条形图
plt.figure(figsize=(8, 6))
sns.barplot(x=mae_labels, y=mae_values, palette='coolwarm')

# 添加标题和标签
plt.title('Model Performance: Original vs Noised Data',font=Path('palatino.ttf'), fontsize=16)
plt.ylabel('Mean Absolute Error (MAE)', font=Path('palatino.ttf'), fontsize=14)
plt.xlabel('Data Type',font=Path('palatino.ttf'), fontsize=14)
plt.savefig('model_performance.pdf', format='pdf')
# 展示图表
plt.show()
