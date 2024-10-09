def test1(num:int, tar:int):
    array = [0] * num
    for i in range(0, num):
       array[i] = int(input())
    res = 0
    left, right = 0, num - 1
    while left < right:
        if array[left] + array[right] == tar:
            res += 1
            left += 1
            right -= 1
        elif array[left] + array[right] < tar:
            left += 1
        else:
            right -= 1
    return res



myinput = input().split()
num = int(myinput[0])
tar = int(myinput[1])
result = test1(num, tar)
print(result)