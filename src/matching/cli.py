import argparse
from typing import List
from .matcher import match_lines
from .similarity import get_context

try:
    from ..preprocessing.normalize import preprocess_file as _pp  # type: ignore
except Exception:
    _pp = None

def _read_lines(path: str) -> List[str]:
    if _pp is not None:
        return _pp(path)
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = " ".join(ln.strip().split()).lower()
            if s:
                out.append(s)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--w-content", type=float, default=0.6)
    ap.add_argument("--w-context", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    left_lines: List[str] = _read_lines(args.old)
    right_lines: List[str] = _read_lines(args.new)

    res = match_lines(
        left_lines, right_lines,
        candidates=None,
        get_context_fn=lambda L, i, r: get_context(L, i, r, include_center=False),
        w_content=args.w_content, w_context=args.w_context,
        threshold=args.threshold, radius=args.radius,
        all_pairs_if_none=True, tie_break="stable",
    )

    rows = [f"{i+1}->{j+1}" for i, j in sorted(res.left_to_right.items())]

    if args.out:
        with open(args.out, "w", encodin)
