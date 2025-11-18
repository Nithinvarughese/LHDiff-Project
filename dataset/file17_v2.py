import socket, time

start = time.time()
host = input("Host to scan: ").strip()
ip_addr = socket.gethostbyname(host)
print("Starting scan on:", ip_addr)

for port in range(50, 500):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    if s.connect_ex((ip_addr, port)) == 0:
        print(f"Port {port}: OPEN")
    s.close()

print("Time taken:", round(time.time() - start, 2), "seconds")
