def count_pairs(nums, target):
    count = 0
    left, right = 0, len(nums) - 1
    while left < right:
        if nums[left] + nums[right] == target:
            count += 1
            left += 1
            right -= 1
        elif nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
    return count

# 读取输入
n, target = map(int, input().split())
nums = [int(input()) for _ in range(n)]

# 计算并输出结果
result = count_pairs(nums, target)
print(result)
