# def min_ava_id(num: int, ids:list[int]) -> int:
#     """最小可用ID"""
#     if num == 0:
#         return 1
#     ids.sort()
#     for i in range(num):
#         if ids[i] != i+1:
#             return i+1
#     return num+1

# num = int(input().strip())
# ids = list(map(int, input().strip().split()))
# print(min_ava_id(num, ids))

def min_ava_id(num: int, ids: list[int]) -> int:
    """最小可用ID"""
    if num == 0:
        return 1
    ids.sort()
    for i in range(num):
        if ids[i] != i + 1:
            return i + 1
    return num + 1

num = int(input().strip())
ids = list(map(int, input().strip().split()))
print(min_ava_id(num, ids))

