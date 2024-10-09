from datetime import datetime, timedelta
start_date  = datetime(2023,1,1)
end_date = datetime(2024,1,1)
delta = timedelta(days=1)
while start_date<=end_date:
    y,m,d = start_date.year,start_date.month,start_date.day
    print("%02d-%02d-%02d",(y,m,d))
    start_date += delta


