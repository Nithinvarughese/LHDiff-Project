import argparse
from typing import List

from .matcher import match_lines

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Line matcher component (content + context similarity)."
    )
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--w-content", type=float, default=0.6)
    ap.add_argument("--w-context", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    if args.w_content < 0 or args.w_context < 0:
        raise SystemExit("Weights must be non-negative.")
    if args.w_content + args.w_context == 0:
        raise SystemExit("At least one of w-content or w-context must be > 0.")
    left_lines = _read_lines(args.old)
    right_lines = _read_lines(args.new)
    res = match_lines(
        left_lines,
        right_lines,
        w_content=args.w_content,
        w_context=args.w_context,
        threshold=args.threshold,
        radius=args.radius,
    )
    rows = [f"{i + 1}->{j + 1}" for i, j in sorted(res.left_to_right.items())]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    if args.print or not args.out:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
