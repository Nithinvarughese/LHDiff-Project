from typing import List, Tuple, Dict
def fields(line: str) -> List[str]:
    out=[]; buf=""; q=False
    for ch in line:
        if ch=='"':
            q=not q
        elif ch==',' and not q:
            out.append(buf); buf=""
        else:
            buf+=ch
    out.append(buf)
    return out
def load(text: str) -> List[Tuple[str,int,float]]:
    out=[]
    for raw in text.splitlines():
        if not raw.strip():
            continue
        cols=fields(raw.strip())
        n=cols[0].strip(); q=int(cols[1].strip()); p=float(cols[2].strip())
        out.append((n,q,p))
    return out
def totals(rows: List[Tuple[str,int,float]]) -> Dict[str,float]:
    m: Dict[str,float]={}
    for n,q,p in rows:
        v=q*p
        m[n]=m.get(n,0.0)+v
    return m
def topk(m: Dict[str,float], k: int) -> List[Tuple[str,float]]:
    it=list(m.items()); it.sort(key=lambda x:(-x[1],x[0]))
    return it[:k]
def report(text: str, k: int=5) -> List[str]:
    rows=load(text); m=totals(rows); it=topk(m,k)
    return [f"{n}:{v:.2f}" for n,v in it]
def dump(rows: List[Tuple[str,int,float]]) -> str:
    return "\n".join(f"{n},{q},{p}" for n,q,p in rows)
def scale(text: str, mult: int=2) -> str:
    rows=load(text)
    out=[(n,q*mult,p) for n,q,p in rows]
    return dump(out)
