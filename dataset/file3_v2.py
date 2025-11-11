# file4_v2.py

def reverse_str(s):  # renamed function
    reversed_s = s[::-1]
    return reversed_s

def count_vowels(s):
    vowels = sum(1 for c in s if c.lower() in "aeiou")
    return vowels

def count_consonants(s):
    return sum(1 for c in s if c.isalpha() and c.lower() not in "aeiou")

def is_palindrome(s):
    s_clean = s.replace(" ", "").lower()
    return s_clean == s_clean[::-1]

def string_summary(s):
    print(f"'{s}' has length {len(s)}")  # changed format
