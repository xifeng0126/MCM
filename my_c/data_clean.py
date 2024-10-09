import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 读取CSV文件
csv_file_path = '2022-wimbledon-1101.csv'
# csv_file_path = 'back.csv'
data = pd.read_csv(csv_file_path)

# # # 将 'AD' 替换为 '50' 并重新分配给相应的列
# data['p1_score'] = data['p1_score'].str.replace('AD', '50')
# data['p2_score'] = data['p2_score'].str.replace('AD', '50')
# 将15替换为1


data['P1Score'] = data['P1Score'].replace(15, 1)
data['P2Score'] = data['P2Score'].replace(15, 1)
data['P1Score'] = data['P1Score'].replace(30, 2)
data['P2Score'] = data['P2Score'].replace(30, 2)
data['P1Score'] = data['P1Score'].replace(40, 3)
data['P2Score'] = data['P2Score'].replace(40, 3)
data['P1Score'] = data['P1Score'].replace('AD', '4')
data['P2Score'] = data['P2Score'].replace('AD', '4')

# 将修改后的数据保存回CSV文件
data.to_csv('2022-wimbledon-1101.csv', index=False)