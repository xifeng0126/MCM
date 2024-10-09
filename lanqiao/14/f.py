ah,al,bh,bl = map(int,input().split())
arr = []
for i in range(ah):
    arr.append(list(map(int,input().split())))
num = (ah-bh+1)*(al-bl+1)
size = bh * bl
vals = [0]*num
h=0
l=0
f=0
while h<=ah-bh:
    l=0
    while l<=al-bl:
        tmp = [0]*size
        for i in range(bh):
            for j in range(bl):
                tmp[i*bl+j]=arr[h+i][l+j]
        tmp.sort()
        vals[f] = tmp[0]*tmp[-1]
        f+=1
        l+=1
    h+=1
ans = sum(vals)%998244353
print(ans)