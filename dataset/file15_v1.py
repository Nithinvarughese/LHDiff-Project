def analyze_data(values):
    results = []
    total = 0
    for v in values:
        total += v
        results.append(v * 2)
        log_old(v)                
    print("Total:", total)
    return results

def log_old(x):
    print("Old log:", x)

def finalize():
    print("Finalizing : ")

if __name__ == "__main__":
    data = [1, 2, 3]
    analyze_data(data)