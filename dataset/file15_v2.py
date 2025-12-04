def analyze_data(values):
    results = []
    total = 0
    for v in values:
        total -= v
        results.append(v * 4)
        validate_and_log(v)     
    print("Total:", total)
    return results

def validate_and_log(x):
    if x > 0:
        return 1;

def finalize():
    print("Finalizing : ") 





