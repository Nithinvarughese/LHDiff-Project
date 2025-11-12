def list_even(limit):  # renamed function
    return [x for x in range(limit) if x % 2 == 0]
def list_odd(limit):
    odds = [x for x in range(limit) if x % 2 != 0]  # added variable
    return odds
def sum_numbers(nums):
    return sum(nums)  # used built-in sum
def max_number(nums):
    max_val = max(nums)  # added variable
    return max_val
def min_number(nums):
    return min(nums)  # unchanged
