from itertools import *
s=[1,2,3]#序列
#l为排列组合的长度:从序列里面取几个出来
l=2
for i in permutations(s,l): #排列
   	print(i)
print("组合")
for i in combinations(s,l): #组合
   	print(i)
