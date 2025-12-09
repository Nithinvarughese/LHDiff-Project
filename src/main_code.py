import hashlib
import re
import argparse
from collections import Counter
from math import sqrt
from typing import List, Dict, Tuple, Optional
import difflib

def normalize_line(line: str) -> str:
    line = line.strip().lower()
    return " ".join(line.split())

def preprocess_lines(lines: List[str]) -> List[str]:
    return [normalize_line(line) for line in lines if line.strip()]

def read_and_preprocess(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return preprocess_lines(f.readlines())

def simhash(text: str, hash_bits: int = 64) -> int:
    tokens = text.split()
    v = [0] * hash_bits
    for token in tokens:
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        for i in range(hash_bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] >= 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming_distance(h1: int, h2: int) -> int:
    x = h1 ^ h2
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist

def generate_candidates(old_lines: List[str], new_lines: List[str], k: int = 15) -> Dict[int, List[int]]:
    candidates = {}
    old_hashes = [simhash(line) for line in old_lines]
    new_hashes = [simhash(line) for line in new_lines]
    for i, h1 in enumerate(old_hashes):
        dists = []
        for j, h2 in enumerate(new_hashes):
            d = hamming_distance(h1, h2)
            dists.append((j, d))
        dists.sort(key=lambda x: x[1])
        candidates[i] = [idx for idx, _ in dists[:k]]
    return candidates

def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_c = curr[j-1] + 1
            delete_c = prev[j] + 1
            replace_c = prev[j-1] + (ca != cb)
            curr.append(min(insert_c, delete_c, replace_c))
        prev = curr
    return prev[-1]

def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    d = levenshtein_distance(a, b)
    m = max(len(a), len(b))
    return 1 - (d / m) if m else 1.0

def get_context(lines: List[str], index: int, r: int = 4) -> List[str]:
    lo = max(0, index - r)
    hi = min(len(lines), index + r + 1)
    return lines[lo:index] + lines[index+1:hi]

def bow_vector(lines: List[str]) -> Counter:
    c = Counter()
    for line in lines:
        for t in line.split():
            c[t] += 1
    return c

def cosine_similarity(v1: Counter, v2: Counter) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(v1[t] * v2.get(t, 0) for t in v1)
    if dot == 0:
        return 0.0
    n1 = sqrt(sum(c*c for c in v1.values()))
    n2 = sqrt(sum(c*c for c in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0

def context_similarity(c1: List[str], c2: List[str]) -> float:
    return cosine_similarity(bow_vector(c1), bow_vector(c2))

def detect_unchanged_lines(old_lines: List[str], new_lines: List[str]) -> Tuple[Dict[int, int], set, set]:
    m = difflib.SequenceMatcher(None, old_lines, new_lines)
    mapping = {}
    used_old = set()
    used_new = set()
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag == 'equal':
            for oi, ni in zip(range(i1, i2), range(j1, j2)):
                mapping[oi] = ni
                used_old.add(oi)
                used_new.add(ni)
    return mapping, used_old, used_new

class LHDiffMatcher:
    def __init__(self, old_lines, new_lines, w_content, w_context, threshold, radius, k):
        self.old_lines = old_lines
        self.new_lines = new_lines
        self.w_content = w_content
        self.w_context = w_context
        self.threshold = threshold
        self.radius = radius
        self.k = k
        self.old_ctx = [get_context(old_lines, i, radius) for i in range(len(old_lines))]
        self.new_ctx = [get_context(new_lines, i, radius) for i in range(len(new_lines))]
        self.cache = {}

    def compute_similarity(self, oi, ni):
        key = (oi, ni)
        if key in self.cache:
            return self.cache[key]
        c_sim = normalized_levenshtein(self.old_lines[oi], self.new_lines[ni])
        x_sim = context_similarity(self.old_ctx[oi], self.new_ctx[ni])
        score = self.w_content * c_sim + self.w_context * x_sim
        self.cache[key] = score
        return score

    def match_with_candidates(self, candidates, used_old, used_new):
        mapping = {}
        owner = {}
        for oi in range(len(self.old_lines)):
            if oi in used_old:
                continue
            if oi not in candidates or not candidates[oi]:
                continue
            best = None
            best_score = self.threshold
            for ni in candidates[oi]:
                if ni in used_new:
                    continue
                s = self.compute_similarity(oi, ni)
                if s > best_score:
                    best = ni
                    best_score = s
            if best is None:
                continue
            if best in owner:
                prev_oi = owner[best]
                prev_s = self.compute_similarity(prev_oi, best)
                if best_score > prev_s:
                    del mapping[prev_oi]
                    mapping[oi] = best
                    owner[best] = oi
            else:
                mapping[oi] = best
                owner[best] = oi
        return mapping

def detect_splits(old_lines, new_lines, mapping, used_old, used_new):
    splits = {}
    for oi in range(len(old_lines)):
        if oi in used_old:
            continue
        old_clean = re.sub(r'\s+', '', old_lines[oi])
        for start in range(len(new_lines)):
            if start in used_new:
                continue
            combined = ''
            idxs = []
            for ni in range(start, len(new_lines)):
                if ni in used_new:
                    break
                combined += re.sub(r'\s+', '', new_lines[ni])
                idxs.append(ni)
                sim = normalized_levenshtein(old_clean, combined)
                if sim > 0.8 and len(idxs) > 1:
                    splits[oi] = idxs
                    used_old.add(oi)
                    used_new.update(idxs)
                    break
                if len(combined) > len(old_clean) * 1.5:
                    break
            if oi in splits:
                break
    return splits

def lhdiff(old_file, new_file, w_content=0.6, w_context=0.4, threshold=0.55, radius=4, k=15, detect_splits_flag=True):
    old_lines = read_and_preprocess(old_file)
    new_lines = read_and_preprocess(new_file)
    unchanged, used_old, used_new = detect_unchanged_lines(old_lines, new_lines)
    candidates = generate_candidates(old_lines, new_lines, k=k)
    for oi in candidates:
        if oi not in used_old:
            candidates[oi] = [j for j in candidates[oi] if j not in used_new]
    matcher = LHDiffMatcher(old_lines, new_lines, w_content, w_context, threshold, radius, k)
    changed = matcher.match_with_candidates(candidates, used_old, used_new)
    final = {}
    for oi, ni in unchanged.items():
        final[oi] = [ni]
    for oi, ni in changed.items():
        final[oi] = [ni]
        used_old.add(oi)
        used_new.add(ni)
    if detect_splits_flag:
        splits = detect_splits(old_lines, new_lines, final, used_old, used_new)
        final.update(splits)
    result = {}
    for oi, nis in final.items():
        result[oi + 1] = [n + 1 for n in nis]
    return result

def format_mapping(m):
    out = []
    for oi in sorted(m.keys()):
        nis = m[oi]
        if len(nis) == 1:
            out.append(f"{oi}->{nis[0]}")
        else:
            out.append(f"{oi}->{','.join(str(n) for n in nis)}")
    return out

def save_mapping(m, output_file):
    lines = format_mapping(m)
    with open(output_file, 'w') as f:
        f.write("\n".join(lines) + "\n")

def print_statistics(mapping, old_file, new_file):
    with open(old_file, 'r') as f:
        old_lines = [l for l in f if l.strip()]
    with open(new_file, 'r') as f:
        new_lines = [l for l in f if l.strip()]
    total_old = len(old_lines)
    total_new = len(new_lines)
    mapped = len(mapping)
    splits = sum(1 for v in mapping.values() if len(v) > 1)
    print("\n" + "="*60)
    print("LHDiff Statistics")
    print("="*60)
    print(f"Old file lines:        {total_old}")
    print(f"New file lines:        {total_new}")
    print(f"Mapped lines:          {mapped}")
    print(f"Unmapped old lines:    {total_old - mapped}")
    print(f"Split lines detected:  {splits}")
    print(f"Mapping coverage:      {mapped/total_old*100:.1f}%")
    print("="*60 + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old', required=True)
    p.add_argument('--new', required=True)
    p.add_argument('--output', '-o')
    p.add_argument('--threshold', type=float, default=0.55)
    p.add_argument('--w-content', type=float, default=0.6)
    p.add_argument('--w-context', type=float, default=0.4)
    p.add_argument('--radius', type=int, default=4)
    p.add_argument('--k', type=int, default=15)
    p.add_argument('--no-splits', action='store_true')
    p.add_argument('--stats', action='store_true')
    args = p.parse_args()

    print("\nRunning LHDiff...")
    print(f"Old file: {args.old}")
    print(f"New file: {args.new}")

    try:
        m = lhdiff(
            args.old, args.new,
            w_content=args.w_content,
            w_context=args.w_context,
            threshold=args.threshold,
            radius=args.radius,
            k=args.k,
            detect_splits_flag=not args.no_splits
        )

        if args.stats:
            print_statistics(m, args.old, args.new)

        result = format_mapping(m)

        if args.output:
            save_mapping(m, args.output)
            print(f"Mapping saved to: {args.output}")
        else:
            print("\nLine Mapping:")
            print("="*60)
            for line in result:
                print(line)

        print("\nLHDiff completed successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
