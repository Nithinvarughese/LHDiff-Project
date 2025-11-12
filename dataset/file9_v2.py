from typing import List, Tuple, Dict
def parse_task(line: str) -> Tuple[str,int]:
    p=line.split(":"); return (p[0], int(p[1]))
def load(text: str) -> List[Tuple[str,int]]:
    out=[]
    for ln in text.splitlines():
        if not ln.strip(): continue
        out.append(parse_task(ln.strip()))
    return out
def schedule(tasks: List[Tuple[str,int]], quantum: int) -> List[str]:
    q=[list(t) for t in tasks]
    out=[]
    t=0
    i=0
    while q:
        name,rem=q[i]
        run=min(quantum,rem)
        rem-=run
        t+=run
        out.append(f"{name}@{t}")
        if rem>0:
            q[i][1]=rem
            i=(i+1)%len(q)
        else:
            q.pop(i)
            if q:
                i%=len(q)
    return out
