import math
n = int(input())
list = list(map(int,input().split()))
sum = 0
for i in range(n):
    sum += math.factorial(list[i])

max = 1
ans = 1
temp = 1
while temp<=sum:
    if sum%temp==0:
        ans = max
    max+=1
    temp=math.factorial(max)
print(ans)
