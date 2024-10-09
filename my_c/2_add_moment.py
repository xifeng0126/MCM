import pandas as pd

df = pd.read_csv('Wimbledon_featured_matches_last.csv')  # 替换为您的文件路径

df['p1_momt'] = df['p1_momt'] + (df['point_victor'] == 1) - (df['point_victor'] == 2)

df['p2_momt'] = df['p2_momt'] + (df['point_victor'] == 2) - (df['point_victor'] == 1)

df.to_csv('Wimbledon_featured_matches_last.csv', index=False)