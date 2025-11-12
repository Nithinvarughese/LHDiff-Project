def add_nums(x, y):  # renamed function
    return x + y
def subtract_numbers(a, b):
    result = a - b  # added variable
    return result
def multiply_numbers(a, b):
    return a * b
def divide_numbers(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
def fibonacci(n):
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[i-1] + seq[i-2])
    return seq
def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(number**0.5)+1):
        if number % i == 0:
            return False
    return True
def sum_of_primes(limit):
    primes = [x for x in range(limit) if is_prime(x)]
    return sum(primes)
def print_summary():
    print("All math functions are ready.")
