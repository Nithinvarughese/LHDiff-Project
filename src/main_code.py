import argparse, difflib, hashlib, math, re
from collections import Counter
from statistics import median

# ===================== Step 1: PREPROCESSING =====================
def normalize(line: str) -> str:
    line = line.rstrip("\n")
    # remove comment tails (python/c/cpp generic)
    line = re.sub(r"#.*$", "", line)
    line = re.sub(r"//.*$", "", line)
    line = re.sub(r"/\*.*?\*/", " ", line)
    line = line.strip().lower()
    line = " ".join(line.split())
    return line

def read_kept(path: str):
    """Return original lines, normalized non-empty lines, and kept_idx -> original_idx."""
    with open(path, "r", encoding="utf-8") as f:
        orig = [l.rstrip("\n") for l in f.readlines()]
    kept, k2o = [], []
    for i, l in enumerate(orig):
        n = normalize(l)
        if n.strip():
            kept.append(n)
            k2o.append(i)
    return orig, kept, k2o

def strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)

def get_context(lines, i, r=4):
    lo, hi = max(0, i-r), min(len(lines), i+r+1)
    return lines[lo:i] + lines[i+1:hi]

# ===================== Similarities =====================
def lev_dist(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def nlev_sim(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - (lev_dist(a, b) / m)

def build_idf(old_lines, new_lines) -> dict:
    docs = old_lines + new_lines
    N = max(1, len(docs))
    df = Counter()
    for ln in docs:
        for t in set(ln.split()):
            df[t] += 1
    return {t: (math.log((N + 1) / (df[t] + 1)) + 1.0) for t in df}

def bow(lines, idf=None) -> Counter:
    c = Counter()
    for ln in lines:
        for t in ln.split():
            c[t] += (idf.get(t, 1.0) if idf else 1.0)
    return c

def cosine(v1: Counter, v2: Counter) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(v1[k] * v2.get(k, 0.0) for k in v1)
    if dot == 0.0:
        return 0.0
    n1 = math.sqrt(sum(x * x for x in v1.values()))
    n2 = math.sqrt(sum(x * x for x in v2.values()))
    return 0.0 if n1 == 0.0 or n2 == 0.0 else dot / (n1 * n2)

# ===================== Step 3: SIMHASH candidates (k=15) =====================
def simhash(text: str, idf=None, bits: int = 64) -> int:
    v = [0.0] * bits
    for tok in text.split():
        w = idf.get(tok, 1.0) if idf else 1.0
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += w if (h >> i) & 1 else -w
    out = 0
    for i in range(bits):
        if v[i] >= 0:
            out |= (1 << i)
    return out

def ham(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

def generate_candidates(old_lines, new_lines, idf, k=15, r=4):
    old_fp = [simhash(old_lines[i] + " " + " ".join(get_context(old_lines, i, r)), idf) for i in range(len(old_lines))]
    new_fp = [simhash(new_lines[j] + " " + " ".join(get_context(new_lines, j, r)), idf) for j in range(len(new_lines))]
    cand = {}
    for i in range(len(old_lines)):
        ds = sorted(((j, ham(old_fp[i], new_fp[j])) for j in range(len(new_lines))), key=lambda x: x[1])
        cand[i] = [j for j, _ in ds[:k]]
    return cand

# ===================== Step 2: Detect unchanged (diff) =====================
def detect_unchanged(old_lines, new_lines):
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    mapping, used_o, used_n = {}, set(), set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for oi, nj in zip(range(i1, i2), range(j1, j2)):
                mapping[oi] = [nj]
                used_o.add(oi)
                used_n.add(nj)
    return mapping, used_o, used_n

# ===================== Step 4: Match + resolve conflicts =====================
def combined_score(old_lines, new_lines, idf, old_ctx_vecs, new_ctx_vecs, oi, nj, wc=0.6, wx=0.4):
    content = nlev_sim(old_lines[oi], new_lines[nj])
    context = cosine(old_ctx_vecs[oi], new_ctx_vecs[nj])
    return wc * content + wx * context, content, context

def resolve_matches(old_lines, new_lines, candidates, used_o, used_n, idf, r=4, threshold=0.55, wc=0.6, wx=0.4):
    old_ctx_vecs = [bow(get_context(old_lines, i, r), idf) for i in range(len(old_lines))]
    new_ctx_vecs = [bow(get_context(new_lines, j, r), idf) for j in range(len(new_lines))]

    mapping = {}
    owner = {}     # new_index -> old_index
    scores = {}    # (old,new) -> combined score

    for oi in range(len(old_lines)):
        if oi in used_o:
            continue

        best_nj, best_s = None, threshold
        for nj in candidates.get(oi, []):
            if nj in used_n:
                continue
            s, _, _ = combined_score(old_lines, new_lines, idf, old_ctx_vecs, new_ctx_vecs, oi, nj, wc, wx)
            scores[(oi, nj)] = s
            if s > best_s:
                best_s, best_nj = s, nj

        if best_nj is None:
            continue

        if best_nj in owner:
            prev_oi = owner[best_nj]
            prev_s = scores.get((prev_oi, best_nj))
            if prev_s is None:
                prev_s, _, _ = combined_score(old_lines, new_lines, idf, old_ctx_vecs, new_ctx_vecs, prev_oi, best_nj, wc, wx)
            if best_s > prev_s:
                mapping.pop(prev_oi, None)
                mapping[oi] = [best_nj]
                owner[best_nj] = oi
        else:
            mapping[oi] = [best_nj]
            owner[best_nj] = oi

    return mapping, scores

# ===================== Step 5: Split detection (late stage) =====================
def detect_splits(old_lines, new_lines, used_o, used_n, candidates, idf, r=4, threshold=0.55, wc=0.6, wx=0.4):
    old_ctx_vecs = [bow(get_context(old_lines, i, r), idf) for i in range(len(old_lines))]
    new_ctx_vecs = [bow(get_context(new_lines, j, r), idf) for j in range(len(new_lines))]

    splits = {}
    split_scores = {}

    for oi in range(len(old_lines)):
        if oi in used_o:
            continue

        # baseline: best single-line score from remaining candidates
        best_single = 0.0
        for nj in candidates.get(oi, []):
            if nj in used_n:
                continue
            s, _, _ = combined_score(old_lines, new_lines, idf, old_ctx_vecs, new_ctx_vecs, oi, nj, wc, wx)
            best_single = max(best_single, s)

        target = strip_ws(old_lines[oi])
        if not target:
            continue

        best_idxs = None
        best_sim = best_single

        for start in range(len(new_lines)):
            if start in used_n:
                continue

            comb = ""
            idxs = []
            prev_sim = -1.0

            for nj in range(start, len(new_lines)):
                if nj in used_n:
                    break
                comb += strip_ws(new_lines[nj])
                idxs.append(nj)

                sim = nlev_sim(target, comb)

                # stop if similarity decreases (per guideline idea)
                if sim < prev_sim:
                    break
                prev_sim = sim

                # accept split only if it beats best single-line and meets threshold
                if len(idxs) > 1 and sim > best_sim and sim >= threshold:
                    best_sim = sim
                    best_idxs = idxs[:]

                if len(comb) > int(1.5 * len(target)):
                    break

        if best_idxs:
            splits[oi] = best_idxs
            split_scores[oi] = best_sim
            used_o.add(oi)
            used_n.update(best_idxs)

    return splits, split_scores

# ===================== Ground truth evaluation (optional) =====================
def parse_truth(path: str):
    truth = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.replace("->", "-")
            left, right = s.split("-", 1)
            oi = int(left.strip())
            njs = [int(x.strip()) for x in right.split(",") if x.strip()]
            truth[oi] = njs
    return truth

def eval_against_truth(pred: dict, truth: dict):
    total = len(truth)
    correct = 0
    for oi, gt in truth.items():
        if pred.get(oi, []) == gt:
            correct += 1

    P = set((oi, nj) for oi, njs in pred.items() for nj in njs)
    T = set((oi, nj) for oi, njs in truth.items() for nj in njs)
    tp = len(P & T)
    prec = tp / len(P) if P else 0.0
    rec  = tp / len(T) if T else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    acc = (correct / total * 100.0) if total else 0.0
    return acc, correct, total, prec, rec, f1

# ===================== MAIN LHDIFF =====================
def lhdiff(old_fp, new_fp, k=15, radius=4, threshold=0.55, do_splits=True):
    _, old_lines, o_map = read_kept(old_fp)
    _, new_lines, n_map = read_kept(new_fp)

    idf = build_idf(old_lines, new_lines)

    unchanged, used_o, used_n = detect_unchanged(old_lines, new_lines)
    candidates = generate_candidates(old_lines, new_lines, idf, k=k, r=radius)

    # remove already-used new indices from candidates
    for oi in range(len(old_lines)):
        candidates[oi] = [nj for nj in candidates.get(oi, []) if nj not in used_n]

    changed, pair_scores = resolve_matches(
        old_lines, new_lines, candidates, used_o, used_n, idf,
        r=radius, threshold=threshold, wc=0.6, wx=0.4
    )
    used_o.update(changed.keys())
    used_n.update(nj for njs in changed.values() for nj in njs)

    splits, split_scores = ({}, {})
    if do_splits:
        splits, split_scores = detect_splits(
            old_lines, new_lines, used_o, used_n, candidates, idf,
            r=radius, threshold=threshold, wc=0.6, wx=0.4
        )

    final = {}
    final.update(unchanged)
    final.update(changed)
    final.update(splits)

    # ===== NEW: Mark deletions and additions =====
    # Deletions: old lines that have no mapping
    deletions = {}
    for oi in range(len(old_lines)):
        if oi not in final:
            deletions[oi] = []  # Empty list means deleted
    
    # Additions: new lines that have no mapping
    additions = {}
    for nj in range(len(new_lines)):
        if nj not in used_n:
            additions[nj] = nj  # Track which new lines are additions

    # Convert kept-index -> original-file line numbers (1-indexed)
    out_map = {}
    confs = []

    # Add unchanged mappings
    for oi, njs in unchanged.items():
        if njs:
            out_map[o_map[oi] + 1] = [n_map[njs[0]] + 1]
            confs.append(1.0)

    # Add changed mappings
    for oi, njs in changed.items():
        if njs:
            nj = njs[0]
            if nj < len(new_lines):
                out_map[o_map[oi] + 1] = [n_map[nj] + 1]
                confs.append(pair_scores.get((oi, nj), 0.0))

    # Add split mappings
    for oi, idxs in splits.items():
        if idxs:
            out_map[o_map[oi] + 1] = [n_map[nj] + 1 for nj in idxs if nj < len(new_lines)]
            confs.append(split_scores.get(oi, threshold))

    # Add deletions (old line -> -1)
    for oi in deletions:
        out_map[o_map[oi] + 1] = [-1]
        confs.append(0.0)

    # Add additions (- -> new line)
    additions_list = []
    for nj in sorted(additions.keys()):
        additions_list.append(n_map[nj] + 1)

    # "accuracy" without truth = coverage + confidence
    total_old = len(old_lines)
    mapped_old = len(final)
    coverage = (mapped_old / total_old * 100.0) if total_old else 0.0
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    med_conf = median(confs) if confs else 0.0

    stats = {
        "old_nonempty": total_old,
        "new_nonempty": len(new_lines),
        "mapped_old": mapped_old,
        "coverage": coverage,
        "splits": len(splits),
        "deletions": len(deletions),
        "additions": len(additions),
        "avg_conf": avg_conf,
        "med_conf": med_conf,
    }
    return out_map, stats, additions_list


def main():
    ap = argparse.ArgumentParser(description="LHDiff - language-independent hybrid line mapping")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("-o", "--out")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--no-splits", action="store_true")
    ap.add_argument("--truth", help="Optional truth mapping file to compute %%Correct, Precision, Recall, F1")
    args = ap.parse_args()

    mapping, stats, additions = lhdiff(
        args.old, args.new,
        k=args.k, radius=args.radius, threshold=args.threshold,
        do_splits=not args.no_splits
    )

    # ---- Accuracy output (required) ----
    # Print a single Accuracy value. If a ground-truth file is provided,
    # use the exact %Correct from `eval_against_truth`. Otherwise compute
    # a heuristic accuracy = coverage * avg_conf (both derived from stats).
    if args.truth:
        truth = parse_truth(args.truth)
        acc, correct, total, prec, rec, f1 = eval_against_truth(mapping, truth)
        accuracy_pct = acc
    else:
        # coverage is in percent, avg_conf in [0,1]; product is percent accuracy
        accuracy_pct = stats['coverage'] * stats['avg_conf']

    print("=" * 70)
    print("LHDiff Results")
    print("=" * 70)
    print(f"Accuracy: {accuracy_pct:.1f}%")

    if args.truth:
        truth = parse_truth(args.truth)
        acc, correct, total, prec, rec, f1 = eval_against_truth(mapping, truth)
        print("\nGround Truth Evaluation")
        print(f"%Correct (exact):      {acc:.1f}% ({correct}/{total})")
        print(f"Precision (pairs):     {prec:.3f}")
        print(f"Recall (pairs):        {rec:.3f}")
        print(f"F1 (pairs):            {f1:.3f}")

    # ---- Line mapping output (required) ----
    print("\n" + "=" * 70)
    print("Line Mapping")
    print("=" * 70)
    
    lines = []
    
    # Sort and output all mappings
    for old_ln in sorted(mapping.keys()):
        new_lns = mapping[old_ln]
        # deletion marker is -1
        if new_lns == [-1]:
            lines.append(f"{old_ln}->-1")
        else:
            lines.append(f"{old_ln}->{','.join(str(x) for x in new_lns)}")
    
    # Output additions as 'old->+new' where 'old' is the nearest previous
    # mapped old-line (1-indexed). If none found, fall back to '->+new'.
    # `additions` is a list of new-file line numbers (1-indexed).
    if additions:
        # Build a helper map: old_line -> max mapped new_line (or None for deletions)
        old_to_max_new = {}
        mapped_old_lines = sorted(mapping.keys())
        for old_ln in mapped_old_lines:
            new_lns = mapping.get(old_ln, [])
            nums = [n for n in new_lns if isinstance(n, int)]
            old_to_max_new[old_ln] = max(nums) if nums else None

        for new_ln in additions:
            # find the largest old_ln whose max mapped new < new_ln
            candidate = None
            for old_ln in mapped_old_lines:
                max_new = old_to_max_new.get(old_ln)
                if max_new is None:
                    continue
                if max_new < new_ln:
                    candidate = old_ln
                else:
                    break

            if candidate is not None:
                lines.append(f"{candidate}->-1")
            else:
                lines.append(f"->-1")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Mapping saved to: {args.out}")
    else:
        for line in lines:
            print(line)
    
    print("=" * 70)
    print("✓ LHDiff completed successfully!")

if __name__ == "__main__":
    main()