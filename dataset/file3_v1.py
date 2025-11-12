def list_even_numbers(limit):
    return [x for x in range(limit) if x % 2 == 0]
def list_odd_numbers(limit):
    return [x for x in range(limit) if x % 2 != 0]
def sum_numbers(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
def max_number(numbers):
    return max(numbers)
def min_number(numbers):
    return min(numbers)
