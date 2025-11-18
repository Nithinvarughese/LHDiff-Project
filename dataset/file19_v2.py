# -*- coding: utf-8 -*-
import time
from calendar import isleap

def is_leap_year(y):
    return isleap(y)

def days_in_month(m, leap):
    if m in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif m in [4, 6, 9, 11]:
        return 30
    elif m == 2 and leap:
        return 29
    else:
        return 28

name = input("Enter your name: ")
age = int(input("Enter your age: "))
now = time.localtime()

years = age
months = years * 12 + now.tm_mon
days = 0

start_year = now.tm_year - years
end_year = start_year + years

for y in range(start_year, end_year):
    days += 366 if is_leap_year(y) else 365

leap = is_leap_year(now.tm_year)
for m in range(1, now.tm_mon):
    days += days_in_month(m, leap)

days += now.tm_mday
print(f"{name}'s age is {years} years or {months} months or {days} days")
