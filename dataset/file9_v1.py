from typing import List, Tuple, Dict
def read_task(line: str) -> Tuple[str,int]:
    p=line.split(":"); return (p[0], int(p[1]))
def parse(text: str) -> List[Tuple[str,int]]:
    out=[]
    for ln in text.splitlines():
        s=ln.strip()
        if not s: continue
        out.append(read_task(s))
    return out
def round_robin(tasks: List[Tuple[str,int]], quantum: int) -> List[str]:
    q=[list(t) for t in tasks]
    out=[]
    t=0
    i=0
    while q:
        name,rem=q[i]
        run=quantum if rem>=quantum else rem
        rem=rem-run
        t=t+run
        out.append(f"{name}@{t}")
        if rem>0:
            q[i][1]=rem
            i=(i+1)%len(q)
        else:
            q.pop(i)
            if q:
                i%=len(q)
    return out
