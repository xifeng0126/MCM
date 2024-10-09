n = int(input())
out = [0]*n
for l in range(n):
    seq = input()
    t = [int(char) for char in seq]
    seq = input()
    s = [int(char) for char in seq]#没必要，直接 s = list(input())
    ans = 0
    if len(s) < 3:
        out[l] = -1
    elif all(x==y for x,y in zip(t,s)):
        out[l] = 0
    else:
        for i in range(len(t)-2):
            if(s[i]==t[i] and s[i+1] != t[i+1] and s[i+2]==t[i+2] and s[i]==s[i+2]):
                ans+=1
        if ans==0:
            ans=-1
        out[l] = ans
for i in range(n):
    print(out[i])
