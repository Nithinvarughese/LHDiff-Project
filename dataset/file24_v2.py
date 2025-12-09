import time
from calendar import isleap

# check leap year
def is_leap(y):
    return isleap(y)

# get days in specific month
def get_month_days(m, is_leap_yr):
    days_map = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap_yr and m == 2:
        return 29
    return days_map[m]

if __name__ == '__main__':
    user_name = input("input your name: ")
    # Convert age immediately
    user_age = int(input("input your age: "))

    local_time = time.localtime(time.time())

    total_months = user_age * 12 + local_time.tm_mon
    total_days = 0

    start_year = int(local_time.tm_year) - user_age
    current_year = start_year + user_age

    # loop years
    for y in range(start_year, current_year):
        if is_leap(y):
            total_days += 366
        else:
            total_days += 365

    is_current_leap = is_leap(local_time.tm_year)
    for m in range(1, local_time.tm_mon):
        total_days += get_month_days(m, is_current_leap)

    total_days += local_time.tm_mday

    print(f"{user_name}'s age is {user_age} years or ", end="")
    print(f"{total_months} months or {total_days} days")