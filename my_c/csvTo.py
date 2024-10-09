import pandas as pd

# 读取CSV文件
csv_file_path = 'Wimbledon_featured_matches.csv'
data = pd.read_csv(csv_file_path)

# 将数据保存为Excel文件
excel_file_path = 'data.xlsx'
data.to_excel(excel_file_path, index=False)
