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
