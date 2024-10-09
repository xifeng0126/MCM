def str_split(str):
    count = 0
    left, right = set(), set(str)
    for i in range(len(str)-1):
        if str[i] not in left:
            left.add(str[i])
        right = set(str[i+1:])
        if len(right) == len(left):
            count += 1
    return count

mystr = input()
print(str_split(mystr))
