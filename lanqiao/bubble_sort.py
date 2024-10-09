def bubble_sort(nums: list[int]):
    n = len(nums)
    for i in range(n):
        for j in range(n-1-i):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]

if __name__ == '__main__':
    nums = [3,2,1,4,5,6,7,8,9,0]
    print(nums)  # [3, 2, 1, 4, 5, 6, 7, 8, 9, 0]
    bubble_sort(nums)
    print(nums)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
