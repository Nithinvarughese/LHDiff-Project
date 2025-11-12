from typing import List
def shape(m: List[List[float]]) -> List[int]:
    return [len(m), len(m[0]) if m else 0]
def add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    r=len(a); c=len(a[0]) if a else 0
    out=[[0.0]*c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            out[i][j]=a[i][j]+b[i][j]
    return out
def mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    r=len(a); c=len(b[0]) if b else 0; k=len(b)
    out=[[0.0]*c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            s=0.0
            for t in range(k):
                s+=a[i][t]*b[t][j]
            out[i][j]=s
    return out
def transpose(a: List[List[float]]) -> List[List[float]]:
    r=len(a); c=len(a[0]) if a else 0
    out=[[0.0]*r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            out[j][i]=a[i][j]
    return out
def flatten(a: List[List[float]]) -> List[float]:
    out=[]
    for row in a:
        out.extend(row)
    return out
