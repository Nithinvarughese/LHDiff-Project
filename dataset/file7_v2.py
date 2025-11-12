from typing import List, Dict
def sanitize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch==" ")
def tokens(s: str) -> List[str]:
    return [t for t in sanitize(s).split() if t]
def ngrams(tokens: List[str], n: int) -> List[str]:
    out=[]
    for i in range(0, len(tokens)-n+1):
        out.append("_".join(tokens[i:i+n]))
    return out
def counts(items: List[str]) -> Dict[str,int]:
    d: Dict[str,int]={}
    for it in items:
        d[it]=d.get(it,0)+1
    return d
def merge_counts(ds: List[Dict[str,int]]) -> Dict[str,int]:
    out: Dict[str,int]={}
    for d in ds:
        for k,v in d.items():
            out[k]=out.get(k,0)+v
    return out
def top_by_prefix(c: Dict[str,int], pref: str, k: int=5) -> List[str]:
    vs=[(k,v) for k,v in c.items() if k.startswith(pref)]
    vs.sort(key=lambda kv:(-kv[1],kv[0]))
    return [x for x,_ in vs[:k]]
def pipeline(texts: List[str], n: int=2) -> Dict[str,int]:
    ds=[]
    for t in texts:
        ts=tokens(t); ns=ngrams(ts,n); ds.append(counts(ns))
    return merge_counts(ds)
