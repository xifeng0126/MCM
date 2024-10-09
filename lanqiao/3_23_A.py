import math
t = 0
num = int(input())
if num== 0:
    print(0)
s = list(map(int,input().split()))
if num == 1:
    print(s[0])
s.sort()
if num == 2:
    print(s[1])
if num>2:
    min = s[0]
    t += int(math.fsum(s))
    t += min*(num-3)
    print(t)

