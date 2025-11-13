"""matcher code"""
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from .similarity import norm_levenshtein, context_cosine, get_context

@dataclass(frozen=True)
class MatchResult:
    left_to_right: Dict[int, int]
    scores: Dict[Tuple[int, int], float]

class Matcher:
    def __init__(
        self,
        left_lines: List[str],
        right_lines: List[str],
        get_context_fn: Callable[[List[str], int, int], List[str]] = get_context,
        w_content: float = 0.6,
        w_context: float = 0.4,
        threshold: float = 0.55,
        radius: int = 4,
    ):
        self.L = left_lines
        self.R = right_lines
        self.get_context = get_context_fn
        self.wc = w_content
        self.wx = w_context
        self.threshold = threshold
        self.radius = radius
        self._ctxL: Dict[int, List[str]] = {}
        self._ctxR: Dict[int, List[str]] = {}

    def _ctx_left(self, i: int) -> List[str]:
        if i not in self._ctxL:
            self._ctxL[i] = self.get_context(self.L, i, self.radius)
        return self._ctxL[i]

    def _ctx_right(self, j: int) -> List[str]:
        if j not in self._ctxR:
            self._ctxR[j] = self.get_context(self.R, j, self.radius)
        return self._ctxR[j]

    def _combined(self, i: int, j: int) -> float:
        c = norm_levenshtein(self.L[i], self.R[j])
        x = context_cosine(self._ctx_left(i), self._ctx_right(j))
        return self.wc * c + self.wx * x

    def score_pairs(
        self,
        candidates: Optional[Dict[int, Iterable[int]]] = None,
        all_pairs_if_none: bool = True,
    ) -> Dict[Tuple[int, int], float]:
        scored: Dict[Tuple[int, int], float] = {}
        if candidates is None and all_pairs_if_none:
            candidates = {i: range(len(self.R)) for i in range(len(self.L))}
        elif candidates is None:
            return scored
        for i, js in candidates.items():
            for j in js:
                s = self._combined(i, j)
                if s >= self.threshold:
                    scored[(i, j)] = s
        return scored

    def greedy_one_to_one(
        self,
        scored_pairs: Dict[Tuple[int, int], float],
        tie_break: str = "stable",
    ) -> Dict[int, int]:
        if tie_break == "stable":
            keyf = lambda it: (-it[1], abs(it[0][0] - it[0][1]), it[0][0], it[0][1])
        elif tie_break == "left":
            keyf = lambda it: (-it[1], it[0][0], it[0][1])
        else:
            keyf = lambda it: (-it[1], it[0][1], it[0][0])
        mapping: Dict[int, int] = {}
        usedL, usedR = set(), set()
        for (i, j), s in sorted(scored_pairs.items(), key=keyf):
            if i in usedL or j in usedR:
                continue
            mapping[i] = j
            usedL.add(i)
            usedR.add(j)
        return mapping

    def match(
        self,
        candidates: Optional[Dict[int, Iterable[int]]] = None,
        all_pairs_if_none: bool = True,
        tie_break: str = "stable",
    ) -> MatchResult:
        scored = self.score_pairs(candidates, all_pairs_if_none)
        mapping = self.greedy_one_to_one(scored, tie_break=tie_break)
        used_scores = {(i, j): scored[(i, j)] for i, j in mapping.items()}
        return MatchResult(mapping, used_scores)

def match_lines(
    left_lines: List[str],
    right_lines: List[str],
    candidates: Optional[Dict[int, Iterable[int]]] = None,
    get_context_fn: Callable[[List[str], int, int], List[str]] = get_context,
    w_content: float = 0.6,
    w_context: float = 0.4,
    threshold: float = 0.55,
    radius: int = 4,
    all_pairs_if_none: bool = True,
    tie_break: str = "stable",
) -> MatchResult:
    m = Matcher(
        left_lines, right_lines,
        get_context_fn=get_context_fn,
        w_content=w_content, w_context=w_context,
        threshold=threshold, radius=radius,
    )
    return m.match(candidates, all_pairs_if_none, tie_break)
