from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
# 定义交叉验证方法
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 使用MAE作为评分指标进行交叉验证
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)

# 打印每次迭代的结果和平均MAE
print("MAE scores for each fold:")
print(-scores)
print(f"Average MAE: {-scores.mean()} with a standard deviation of {scores.std()}")
# 交叉验证MAE分数
cv_mae_scores = [0.20657118, 0.18799686, 0.19072763, 0.1973965, 0.1888233]

# 创建箱形图
plt.figure(figsize=(8, 6))
sns.boxplot(y=cv_mae_scores, palette='pastel')

# 添加水平线表示平均MAE
plt.axhline(y=np.mean(cv_mae_scores), color='r', linestyle='--', label='Average MAE')

# 添加标题和标签
plt.title('Cross-Validation MAE Scores Distribution', font=Path('palatino.ttf'), fontsize=16)
plt.ylabel('Mean Absolute Error (MAE)', font=Path('palatino.ttf'), fontsize=14)
plt.legend()
plt.savefig('performance_02.pdf', format='pdf')
# 展示图表
plt.show()
