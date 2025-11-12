def add_numbers(a, b):
    return a + b
def subtract_numbers(a, b):
    return a - b
def multiply_numbers(a, b):
    return a * b
def divide_numbers(a, b):
    if b == 0:
        return None
    return a / b
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
def fibonacci(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True
def sum_of_primes(limit):
    return sum([x for x in range(limit) if is_prime(x)])
def print_summary():
    print("Math functions loaded.")
