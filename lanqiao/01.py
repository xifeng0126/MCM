from queue import PriorityQueue #优先队列
a=list(map(int,input().split()))
q=PriorityQueue()          #创建数据结构
for i in a:
    q.put(i)               #入队
if q.qsize()!=1:        #队列大小
        a=q.get() 
#出队  
print(a)
print(q.get())