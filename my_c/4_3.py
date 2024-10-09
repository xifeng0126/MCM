import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


df2 = pd.read_csv('2022-wimbledon-1101.csv')


# 计算皮尔逊相关系数
correlation = np.corrcoef(df2['momentum_shift_pred'], df2['momentum_shift'])[0, 1]
print(f"Pearson Correlation Coefficient: {correlation}")

# 计算符号一致性
sign_consistency = np.mean(np.sign(df2['momentum_shift_pred']) == np.sign(df2['momentum_shift']))
print(f"Sign Consistency Ratio: {sign_consistency * 100}%")

# Scatter plot with regression line
sns.set_style("whitegrid")

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
scatter = sns.regplot(x='momentum_shift_pred', y='momentum_shift', data=df2,
                      scatter_kws={'alpha':0.6, 'color': 'blue'}, line_kws={'color': 'red'})

plt.title('Predicted vs. Actual Values',font=Path('palatino.ttf'), fontsize=16)
plt.xlabel('Predicted Value (momentum_shift_pred)', font=Path('palatino.ttf'), fontsize=14)
plt.ylabel('Actual Value (momentum_shift)',font=Path('palatino.ttf'),fontsize=14)

# Optionally, highlight specific points (e.g., outliers)
# plt.annotate('Outlier', xy=(x_value, y_value), xytext=(x_text, y_text),
#              arrowprops=dict(facecolor='black', shrink=0.05))
plt.savefig('regression.pdf', format='pdf')
plt.show()

# plt.figure(figsize=(10, 6))
#
# errors = df2['momentum_shift_pred'] - df2['momentum_shift']
# # 将误差分类为正误差和负误差
# positive_errors = errors[errors >= 0]
# negative_errors = errors[errors < 0]
# # Plot positive errors in green
# sns.histplot(positive_errors, kde=True, color='green', binwidth=0.5, label='Positive Errors')
#
# # Plot negative errors in red
# sns.histplot(negative_errors, kde=True, color='red', binwidth=0.5, label='Negative Errors')
#
# # 绘制平均误差线
# plt.title('Error Distribution', font=Path('palatino.ttf'), fontsize=16)
# plt.xlabel('Prediction Error (momentum_shift_pred - momentum_shift)', font=Path('palatino.ttf'), fontsize=14)
# plt.ylabel('Frequency', font=Path('palatino.ttf'), fontsize=14)
# plt.legend()
# plt.savefig('Error Distribute.pdf', format='pdf')
# plt.show()