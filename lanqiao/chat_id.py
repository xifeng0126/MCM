def find_min_id(N, allocated_ids):
    allocated_ids_set = set(allocated_ids)
    print(allocated_ids_set)
    min_id = 1
    while min_id in allocated_ids_set:
        min_id += 1
    return min_id

# 读取输入
N = int(input())
allocated_ids = list(map(int, input().split()))

# 调用函数找出最小可以分配的ID
min_id = find_min_id(N, allocated_ids)

# 输出结果
print(min_id)
