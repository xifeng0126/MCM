def select_sort(arr):
    for i in range(len(arr)):
        k = i
        for j in range(i+1,len(arr)):
            if arr[j] < arr[k]:
                k = j
        arr[i],arr[k] = arr[k],arr[i]

if __name__ == '__main__':
    arr = [3,2,1,4,5,6,7,8,9,0]
    print(arr)  # [3, 2, 1, 4, 5, 6, 7, 8, 9, 0]
    select_sort(arr)
    print(arr)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]