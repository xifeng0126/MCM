import pandas as pd

# 示例数据，假设df是您的DataFrame
# df = pd.DataFrame({
#     'elapsed_time': ['1:05:30', '0:35:45', '2:15:00', ...]
# })
df = pd.read_csv('Wimbledon_featured_matches_last.csv')

# 将'elapsed_time'列的时间从HH:MM:SS格式转换为秒
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time']).dt.total_seconds()

df.to_csv('Wimbledon_featured_matches_last.csv', index=False)
