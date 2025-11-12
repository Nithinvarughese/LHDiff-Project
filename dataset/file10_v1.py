from typing import List, Tuple
def rect_area(w: float, h: float) -> float:
    return w*h
def circle_area(r: float) -> float:
    return 3.141592653589793*r*r
def rect_peri(w: float, h: float) -> float:
    return 2*(w+h)
def circle_peri(r: float) -> float:
    return 2*3.141592653589793*r
def bounds(points: List[Tuple[float,float]]) -> Tuple[float,float,float,float]:
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    mnx=xs[0]; mny=ys[0]; mxx=xs[0]; mxy=ys[0]
    i=1
    while i<len(xs):
        x=xs[i]; y=ys[i]
        if x<mnx: mnx=x
        if y<mny: mny=y
        if x>mxx: mxx=x
        if y>mxy: mxy=y
        i+=1
    return (mnx,mny,mxx,mxy)
def center(points: List[Tuple[float,float]]) -> Tuple[float,float]:
    sx=0.0; sy=0.0; i=0; n=len(points)
    while i<n:
        x,y=points[i]; sx+=x; sy+=y; i+=1
    return ((sx/n, sy/n) if n else (0.0,0.0))
