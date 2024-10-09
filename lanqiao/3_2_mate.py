def final_days(m:int, k:int) -> int:
    """最终天数"""
    f_days = m
    while m >= k:
        a_dayas = m // k
        m -=  a_dayas * (k-1)
        f_days += a_dayas
    return f_days


m, k = map(int, input().strip().split())
print(final_days(m, k))