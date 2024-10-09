def insert_sort(nums: list[int]):
    n = len(nums)
    for i in range(1,n):
        base = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > base:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = base

if __name__ == '__main__':
    nums = [3,2,1,4,5,6,7,8,9,0]
    print(nums)  # [3, 2, 1, 4, 5, 6, 7, 8, 9, 0]
    insert_sort(nums)
    print(nums)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]