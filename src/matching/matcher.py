from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    ) -> None:
        self.left_lines = left_lines
        self.right_lines = right_lines
        self.get_context_fn = get_context_fn
        self.w_content = w_content
        self.w_context = w_context
        self.threshold = threshold
        self.radius = radius
        self._left_ctx = [
            self.get_context_fn(left_lines, i, radius)
            for i in range(len(left_lines))
        ]
        self._right_ctx = [
            self.get_context_fn(right_lines, i, radius)
            for i in range(len(right_lines))
        ]
        self._score_cache: Dict[Tuple[int, int], float] = {}

    def _pair_score(self, i: int, j: int) -> float:
        key = (i, j)
        if key in self._score_cache:
            return self._score_cache[key]
        s_content = norm_levenshtein(self.left_lines[i], self.right_lines[j])
        s_context = context_cosine(self._left_ctx[i], self._right_ctx[j])
        score = self.w_content * s_content + self.w_context * s_context
        self._score_cache[key] = score
        return score

    def match(
        self,
        candidates: Optional[Dict[int, List[int]]] = None,
        all_pairs_if_none: bool = True,
        tie_break: str = "stable",
    ) -> MatchResult:
        n_left = len(self.left_lines)
        n_right = len(self.right_lines)
        if candidates is None:
            if all_pairs_if_none:
                candidates = {i: list(range(n_right)) for i in range(n_left)}
            else:
                candidates = {}
        left_to_right: Dict[int, int] = {}
        scores: Dict[Tuple[int, int], float] = {}
        right_owner: Dict[int, int] = {}
        for i in range(n_left):
            cand_js = candidates.get(i)
            if not cand_js:
                continue
            best_j: Optional[int] = None
            best_score: float = self.threshold
            for j in cand_js:
                if j < 0 or j >= n_right:
                    continue
                s = self._pair_score(i, j)
                scores[(i, j)] = s
                if s >= best_score:
                    best_score = s
                    best_j = j
            if best_j is None:
                continue
            owner = right_owner.get(best_j)
            if owner is None:
                left_to_right[i] = best_j
                right_owner[best_j] = i
            else:
                if tie_break == "max":
                    old_score = scores.get(
                        (owner, best_j), self._pair_score(owner, best_j)
                    )
                    if best_score > old_score:
                        left_to_right.pop(owner, None)
                        left_to_right[i] = best_j
                        right_owner[best_j] = i
                else:
                    continue
        return MatchResult(left_to_right=left_to_right, scores=scores)


def match_lines(
    left_lines: List[str],
    right_lines: List[str],
    candidates: Optional[Dict[int, List[int]]] = None,
    *,
    get_context_fn: Callable[[List[str], int, int], List[str]] = get_context,
    w_content: float = 0.6,
    w_context: float = 0.4,
    threshold: float = 0.55,
    radius: int = 4,
    all_pairs_if_none: bool = True,
    tie_break: str = "stable",
) -> MatchResult:
    matcher = Matcher(
        left_lines=left_lines,
        right_lines=right_lines,
        get_context_fn=get_context_fn,
        w_content=w_content,
        w_context=w_context,
        threshold=threshold,
        radius=radius,
    )
    return matcher.match(
        candidates=candidates,
        all_pairs_if_none=all_pairs_if_none,
        tie_break=tie_break,
    )