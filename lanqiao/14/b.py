import sys
sys.setrecursionlimit(5000000)
a = [0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5]
b = [2,1,0,3,2,1,0,4,3,2,1,0,5,4,3,2,1,0]
ans = 0
def dfs(n,c1,c2):
    global ans
    if n == 7:
        if not c1 and not c2:
            ans+=1
    else:
        for i in range(len(a)):
            if c1 >= a[i] and c2 >= b[i]:
                dfs(n+1,c1-a[i],c2-b[i])

dfs(0,9,16)
print(ans)