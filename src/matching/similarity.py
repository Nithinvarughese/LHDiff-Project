from collections import Counter
from math import sqrt
from typing import Iterable, List


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def norm_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len


def _bow(lines: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for line in lines:
        for token in line.split():
            counter[token] += 1
    return counter


def _cosine(v1: Counter, v2: Counter) -> float:
    if not v1 or not v2:
        return 0.0
    dot = 0.0
    for term, c1 in v1.items():
        dot += c1 * v2.get(term, 0.0)
    if dot == 0.0:
        return 0.0
    n1 = sqrt(sum(c * c for c in v1.values()))
    n2 = sqrt(sum(c * c for c in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def context_cosine(left_ctx: List[str], right_ctx: List[str]) -> float:
    return _cosine(_bow(left_ctx), _bow(right_ctx))


def get_context(
    lines: List[str],
    i: int,
    radius: int = 4,
    include_center: bool = False,
) -> List[str]:
    if radius < 0:
        radius = 0
    lo = max(0, i - radius)
    hi = min(len(lines), i + radius + 1)
    if include_center:
        return lines[lo:hi]
    return lines[lo:i] + lines[i + 1 : hi]
