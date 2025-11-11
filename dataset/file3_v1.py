# file4_v1.py

def reverse_string(s):
    return s[::-1]

def count_vowels(s):
    return sum(1 for c in s if c.lower() in "aeiou")

def count_consonants(s):
    return sum(1 for c in s if c.isalpha() and c.lower() not in "aeiou")

def is_palindrome(s):
    return s == s[::-1]

def string_summary(s):
    print(f"String: {s}, Length: {len(s)}")
