def max_mixed_doubles_advantage(n, P, Q):
    dp = [[0] * (1 << n) for _ in range(n)]  # dp[i][mask]表示前i个男选手已经被选中，状态为mask时的最大优势

    for mask in range(1 << n):
        for j in range(n):
            if mask & (1 << j):  # 检查第j个男选手是否已被选中
                for i in range(n):
                    if not (mask & (1 << i)):  # 检查第i个女选手是否已被选中
                        dp[j][mask] = max(dp[j][mask], dp[j - 1][mask ^ (1 << j)] + P[j][i] * Q[i][j])
                    else:
                        dp[j][mask] = max(dp[j][mask], dp[j - 1][mask ^ (1 << j)])

    return max(dp[-1])  # 返回最后一个男选手全部被选中时的最大优势

# 读取输入
n = int(input().strip())
P = []
Q = []
for _ in range(n):
    P.append(list(map(int, input().strip().split())))
for _ in range(n):
    Q.append(list(map(int, input().strip().split())))

# 输出结果
print(max_mixed_doubles_advantage(n, P, Q))
