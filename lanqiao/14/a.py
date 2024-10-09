class sa:
    def __init__(self,y,m,d):
        self.y = y
        self.m = m
        self.d = d

months = [0,31,28,31,30,31,30,31,31,30,31,30,31]

def f1(y,m,d):
    if(y%4==0 and y%100) or (y%400==0):
        months[2] = 29
    else:
        months[2] = 28
    d += 1
    if d > months[m]:
        m+=1
        d=1
    if m > 12:
        m=1
        y+=1
    return sa(y,m,d)

t1 = sa(2000,1,1)
t2 = sa(2000000,1,1)
ans = 0

while True:
    if(t1.y % t1.m==0)and(t1.y%t1.d==0):
        ans+=1
    t1=f1(t1.y,t1.m,t1.d)
    if(t1.y==t2.y and t1.m == t2.m and t1.d == t2.d):
        break
    
print(ans)
        
