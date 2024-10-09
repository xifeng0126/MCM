def find_most(cn,a,b,c):
    if cn ==0:
        mylist = [a[i]-b[i]-c[i] for i in range(len(a))]
    elif cn==1:
        mylist = [b[i]-a[i]-c[i] for i in range(len(a))]#不能用两个if语句，否则，第二个else肯定执行
    else:
        mylist = [c[i]-a[i]-b[i] for i in range(len(a))]
    mylist.sort(reverse=True)
    s = 0
    for i in range(len(mylist)):
        s += mylist[i]
        if s <= 0:
            if i==0:
                return -1
            return i
    return len(a)

n = int(input())
a = list(map(int,input().split()))
b = list(map(int,input().split()))
c = list(map(int,input().split()))
res = [find_most(i,a,b,c) for i in range(3)]
print(max(res))