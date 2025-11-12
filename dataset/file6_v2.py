from typing import List, Tuple, Dict
def split_csv(line: str) -> List[str]:
    out=[]; buf=""; i=0; q=False
    while i<len(line):
        ch=line[i]
        if ch=='"':
            q=not q
        elif ch==',' and not q:
            out.append(buf); buf=""
        else:
            buf+=ch
        i+=1
    out.append(buf)
    return out
def parse_rows(text: str) -> List[Tuple[str,int,float]]:
    rows=[]
    for ln in text.splitlines():
        if not ln.strip():
            continue
        cols=split_csv(ln.strip())
        name=cols[0].strip()
        qty=int(cols[1].strip())
        price=float(cols[2].strip())
        rows.append((name,qty,price))
    return rows
def group_total(rows: List[Tuple[str,int,float]]) -> Dict[str,float]:
    d: Dict[str,float]={}
    for name,qty,price in rows:
        val=qty*price
        if name in d:
            d[name]+=val
        else:
            d[name]=val
    return d
def select_top(d: Dict[str,float], k: int) -> List[Tuple[str,float]]:
    items=list(d.items())
    items.sort(key=lambda x:(-x[1],x[0]))
    return items[:k]
def run_report(text: str, k: int=5) -> List[str]:
    rows=parse_rows(text)
    totals=group_total(rows)
    top=select_top(totals,k)
    out=[]
    for name,val in top:
        out.append(f"{name}:{val:.2f}")
    return out
def serialize(rows: List[Tuple[str,int,float]]) -> str:
    lines=[]
    for n,q,p in rows:
        lines.append(f"{n},{q},{p}")
    return "\n".join(lines)
def expand(text: str, mult: int=2) -> str:
    rows=parse_rows(text)
    out=[]
    for n,q,p in rows:
        out.append((n,q*mult,p))
    return serialize(out)
