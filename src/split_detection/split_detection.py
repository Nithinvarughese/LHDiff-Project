import difflib
import sys
import re


# --------------------------------
# STEP 1: PREPROCESS
# --------------------------------
def preprocess(lines):
    result = []
    for line in lines:
        # remove all spaces/tabs and lowercase for matching
        clean_line = re.sub(r"\s+", "", line).lower()
        result.append(clean_line)
    return result


# --------------------------------
# STEP 2: READ FILE
# --------------------------------
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()


# --------------------------------
# STEP 3: FIND UNCHANGED (1 → 1)
# --------------------------------
def find_exact_matches(old, new):

    matcher = difflib.SequenceMatcher(a=old, b=new)
    mapping = {}
    used_new = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for o, n in zip(range(i1, i2), range(j1, j2)):
                mapping[o] = [n]
                used_new.add(n)

    return mapping, used_new


# --------------------------------
# STEP 4: FIND SPLITS (1 → MANY)
# --------------------------------
def find_splits(old, new, mapping, used_new):

    split_map = {}

    for old_index, old_line in enumerate(old):

        if old_index in mapping:
            continue

        for start in range(len(new)):

            if start in used_new:
                continue

            combined = ""
            indices = []

            for j in range(start, len(new)):

                if j in used_new:
                    break

                combined += new[j]
                indices.append(j)

                # Check: does combined new lines reconstruct old line?
                if old_line in combined or combined in old_line:
                    split_map[old_index] = indices[:]   # copy list
                    used_new.update(indices)
                    break

    return split_map


# --------------------------------
# STEP 5: FIND BEST REMAINING MATCHES
# --------------------------------
def find_best_matches(old, new, mapping, used_new):

    for old_index, old_line in enumerate(old):

        if old_index in mapping:
            continue

        best_score = 0
        best_match = None

        for new_index, new_line in enumerate(new):

            if new_index in used_new:
                continue

            score = difflib.SequenceMatcher(None, old_line, new_line).ratio()

            if score > best_score:
                best_score = score
                best_match = new_index

        if best_match is not None:
            mapping[old_index] = [best_match]
            used_new.add(best_match)

    return mapping


# --------------------------------
# MAIN FUNCTION
# --------------------------------
def split_detection(old_file, new_file):

    print("\n--- SPLIT DETECTION STARTED ---")

    old_raw = read_file(old_file)
    new_raw = read_file(new_file)

    old = preprocess(old_raw)
    new = preprocess(new_raw)

    # Step 1: Unchanged lines
    mapping, used_new = find_exact_matches(old, new)

    # Step 2: Split detection (1 → many)
    split_map = find_splits(old, new, mapping, used_new)
    mapping.update(split_map)

    # Step 3: Remaining best matches
    mapping = find_best_matches(old, new, mapping, used_new)

    print("\n--- FINAL LINE MAPPING (OLD → NEW) ---\n")

    for old_index in sorted(mapping.keys()):
        new_lines = mapping[old_index]
        result = ",".join(str(n + 1) for n in new_lines)
        print(f"{old_index + 1} -> {result}")

    print("\n--- SPLIT DETECTION COMPLETE ---")


# --------------------------------
# RUN FROM TERMINAL
# --------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 split_detection.py <old_file> <new_file>")
        sys.exit(1)

    old_file = sys.argv[1]
    new_file = sys.argv[2]

    split_detection(old_file, new_file)
