import difflib
import os
import argparse
from typing import List


class LineMappingDetector:
    """
    Produces a line-to-line mapping:
        1 -> 1
        2 -> 3
        3 -> deleted
        - -> 4
    """

    def __init__(self, old_lines: List[str], new_lines: List[str]):
        self.old = old_lines
        self.new = new_lines

    def generate_line_mapping(self) -> List[str]:
        sm = difflib.SequenceMatcher(None, self.old, self.new)
        opcodes = sm.get_opcodes()

        mapping = []
        new_mapped = set()

        for tag, i1, i2, j1, j2 in opcodes:

            if tag == "equal":
                for o, n in zip(range(i1, i2), range(j1, j2)):
                    mapping.append(f"{o+1} -> {n+1}")
                    new_mapped.add(n+1)

            elif tag == "replace":
                for o in range(i1, i2):
                    mapping.append(f"{o+1} -> modified")
                for n in range(j1, j2):
                    new_mapped.add(n+1)

            elif tag == "delete":
                for o in range(i1, i2):
                    mapping.append(f"{o+1} -> deleted")

            elif tag == "insert":
                continue

        # Add new lines not mapped
        for i in range(1, len(self.new) + 1):
            if i not in new_mapped:
                mapping.append(f"- -> {i}")

        return mapping


def run_mapping(old_path: str, new_path: str) -> List[str]:
    with open(old_path, "r") as f:
        old_lines = f.readlines()
    with open(new_path, "r") as f:
        new_lines = f.readlines()

    detector = LineMappingDetector(old_lines, new_lines)
    return detector.generate_line_mapping()


def run_all(dataset_folder: str):
    files = os.listdir(dataset_folder)
    v1_files = [f for f in files if "_v1.py" in f]

    for v1 in v1_files:
        base = v1.replace("_v1.py", "")
        v2 = f"{base}_v2.py"

        v1_path = os.path.join(dataset_folder, v1)
        v2_path = os.path.join(dataset_folder, v2)

        if not os.path.exists(v2_path):
            print(f"Missing: {v2}")
            continue

        print(f"\n=== {base} ===")
        mapping = run_mapping(v1_path, v2_path)
        for m in mapping:
            print(m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    # Run ALL dataset pairs
    if args.all:
        run_all("dataset")      # <-- FIXED PATH
        exit()

    # Run single file pair
    if args.file:
        base = args.file
        v1 = f"dataset/{base}_v1.py"   # <-- FIXED PATH
        v2 = f"dataset/{base}_v2.py"   # <-- FIXED PATH

        if not os.path.exists(v1):
            print(f"❌ Not found: {v1}")
            exit()
        if not os.path.exists(v2):
            print(f"❌ Not found: {v2}")
            exit()

        print(f"\n=== Mapping for {base} ===")
        mapping = run_mapping(v1, v2)
        for m in mapping:
            print(m)

    else:
        print("Usage:\n  --file file1\n  --all")
