def find_num(init):
    res = str(init)
    for _ in range(1, init):
        temp = res[0]
        count = 0
        newres = ''
        for i in res:
            if i == temp:
                count += 1
            else:
                newres = newres + str(count) + str(temp)
                temp = i
                count = 1
        res = newres + str(count) + str(temp)
    return res

first = int(input())
print(find_num(first))
