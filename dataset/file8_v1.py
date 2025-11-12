from typing import List
def dims(m: List[List[float]]) -> List[int]:
    return [len(m), len(m[0]) if m else 0]
def plus(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    r=len(a); c=len(a[0]) if a else 0
    x=[[0.0]*c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            x[i][j]=a[i][j]+b[i][j]
    return x
def dot(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    r=len(a); c=len(b[0]) if b else 0; k=len(b)
    x=[[0.0]*c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            s=0.0; t=0
            while t<k:
                s+=a[i][t]*b[t][j]; t+=1
            x[i][j]=s
    return x
def t(a: List[List[float]]) -> List[List[float]]:
    r=len(a); c=len(a[0]) if a else 0
    x=[[0.0]*r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            x[j][i]=a[i][j]
    return x
def ravel(a: List[List[float]]) -> List[float]:
    out=[]
    i=0
    while i<len(a):
        out.extend(a[i])
        i+=1
    return out
