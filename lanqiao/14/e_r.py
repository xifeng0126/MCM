#模拟法
n = int(input())
for _ in range(n):
    ans = 0
    t = list(input())
    s = list(input())
    for i in range(1,len(s)-1):
        if t[i] != s[i]:
            if s[i-1] == s[i+1] and s[i-1]!=s[i]:
                s[i] = t[i]
                ans+=1
    if t==s:
        print(ans)
    else:
        print(-1)

    