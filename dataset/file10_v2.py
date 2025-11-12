from typing import List, Tuple
def area_rect(w: float, h: float) -> float:
    return w*h
def area_circle(r: float) -> float:
    return 3.141592653589793*r*r
def peri_rect(w: float, h: float) -> float:
    return 2*(w+h)
def peri_circle(r: float) -> float:
    return 2*3.141592653589793*r
def bbox(points: List[Tuple[float,float]]) -> Tuple[float,float,float,float]:
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
def centroid(points: List[Tuple[float,float]]) -> Tuple[float,float]:
    sx=0.0; sy=0.0
    for x,y in points:
        sx+=x; sy+=y
    n=len(points)
    return (sx/n, sy/n) if n else (0.0,0.0)
