from socket import *
import time

startTime = time.time()

if __name__ == '__main__':
    target = input('Enter the host to be scanned: ')
    t_ip = gethostbyname(target)
    print('Starting scan on host:', t_ip)

    for port in range(50, 500):
        s = socket(AF_INET, SOCK_STREAM)
        connection = (t_ip, port)     # split part 1
        conn = s.connect_ex(connection)  # split part 2

        if conn == 0:
            print(f'Port {port}: OPEN')
        s.close()

print('Time taken:', round(time.time() - startTime, 2))
