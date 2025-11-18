import time

def timer_countdown(sec):
    try:
        sec = int(sec)
    except ValueError:
        print("Enter an integer number of seconds.")
        return

    while sec > 0:
        mins, secs = divmod(sec, 60)
        print(f"{mins:02d}:{secs:02d}", end="\r")
        time.sleep(1)
        sec -= 1

    print("Timer completed!   ")

if __name__ == "__main__":
    secs = input("Seconds: ").strip()
    timer_countdown(secs)
