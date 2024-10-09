def calculate_days(M, K):
    total_days = M  # 初始总天数等于初始金额M
    while M >= K:  # 当剩余金额大于等于K时，可以继续获赠1元
        free_money = M // K  # 计算可以获赠的金额
        total_days += free_money  # 总天数增加获赠的金额
        M = M - free_money * K + free_money  # 更新剩余金额，扣除K元后再加上获赠的金额
    return total_days

# 读取输入
M, K = map(int, input().split())

# 调用函数计算可以用卫星通话的天数
result = calculate_days(M, K)

# 输出结果
print(result)
