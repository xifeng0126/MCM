def partition(nums, left, right):
    i, j = left, right
    while i<j:
        while i < j and nums[left] <= nums[j]:
            j -= 1
        while i < j and nums[i] <= nums[left]:
            i += 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i], nums[left] = nums[left], nums[i]
    return i

def quick_sort(nums, left, right):
    if left >= right:
        return
    pivot = partition(nums, left, right)
    quick_sort(nums, left, pivot-1)
    quick_sort(nums, pivot+1, right)

if __name__ == '__main__':
    nums = [2,4,1,0,3,5]
    print(nums)  # [3, 2, 1, 4, 5, 6, 7, 8, 9, 0]
    quick_sort(nums, 0, len(nums)-1)
    print(nums)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]