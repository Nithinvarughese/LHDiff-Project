def square_number(n):  # renamed function
    return n * n
def cube(n):
    result = n ** 3  # added variable
    return result
def power(base, exp):  # renamed parameter
    return base ** exp
def list_powers(n):
    return [i**2 for i in range(n+1)]  # include n
def print_numbers(n):
    for i in range(n):
        print(f"Number: {i}")  # changed print format
