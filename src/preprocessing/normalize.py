def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
def clean_line(line):
        line = line.strip()
        line = line.lower()
        parts = line.split()
        return " ".join(parts)
def preprocess_lines(lines):
        normalized = []
        for line in lines:
            normalized.append(clean_line(line))
        return normalized
def preprocess_file(path):
        raw = read_file(path)
        return preprocess_lines(raw)
lines1 = preprocess_file("dataset/file_v1.py")
lines2 = preprocess_file("dataset/file_v2.py")
print("Normalized lines from file1_v1.py:")
for i, line in enumerate(lines1, start=1):
        print(f"{i}: {line}")
print("\nNormalized lines from file1_v2.py:")
for i, line in enumerate(lines2, start=1):
        print(f"{i}: {line}")