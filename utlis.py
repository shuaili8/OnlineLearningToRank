def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)