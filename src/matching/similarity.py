from collections import Counter
from math import sqrt
from typing import Iterable, List

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            dele = curr[j - 1] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def norm_levenshtein(a: str, b: str) -> float:
    if not a and not b: return 1.0
    d = levenshtein(a, b)
    return 1.0 - (d / max(len(a), len(b), 1))

def _bow(lines: Iterable[str]) -> Counter:
    c = Counter()
    for ln in lines:
        for tok in ln.split():
            c[tok] += 1
    return c

def _cosine(c1: Counter, c2: Counter) -> float:
    if not c1 or not c2: return 0.0
    dot = sum(c1[k] * c2.get(k, 0) for k in c1)
    n1 = sqrt(sum(v * v for v in c1.values()))
    n2 = sqrt(sum(v * v for v in c2.values()))
    return (dot / (n1 * n2)) if n1 and n2 else 0.0

def context_cosine(left_ctx: List[str], right_ctx: List[str]) -> float:
    return _cosine(_bow(left_ctx), _bow(right_ctx))

def get_context(lines: List[str], i: int, radius: int = 4, include_center: bool = False) -> List[str]:
    lo = max(0, i - radius)
    hi = min(len(lines), i + radius + 1)
    return lines[lo:hi] if include_center else (lines[lo:i] + lines[i + 1:hi])
