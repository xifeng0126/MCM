dp = [[0 for _ in range(100005)] for _ in range(105)]

def main():
    n = int(input())
    nums = list(map(int,input().split()))
    dp[0][0] = 1
    for i in range(1, n+1):
        a = nums[i-1]
        for j in range(100001):
            if dp[i-1][j]:
                dp[i][j] = 1
                dp[i][j + a] = 1
                dp[i][abs(j - a)] = 1
    ans = 0
    for i in range(1, 100001):
        if dp[n][i]:
            ans += 1
    print(ans)

if __name__ == "__main__":
    main()
