from typing import List, Dict, Tuple
def clean(s: str) -> str:
    return "".join(ch.lower() for ch in s if ("a"<=ch<="z") or ("0"<=ch<="9") or ch==" ")
def split_words(s: str) -> List[str]:
    s=clean(s); parts=[p for p in s.split() if p]
    return parts
def build_ngrams(parts: List[str], n: int) -> List[str]:
    out=[]; i=0; L=len(parts)
    while i+n<=L:
        out.append("_".join(parts[i:i+n])); i+=1
    return out
def tally(items: List[str]) -> Dict[str,int]:
    out: Dict[str,int]={}
    for x in items:
        out[x]=out.get(x,0)+1
    return out
def reduce_sum(frags: List[Dict[str,int]]) -> Dict[str,int]:
    out: Dict[str,int]={}
    for d in frags:
        for k,v in d.items():
            out[k]=out.get(k,0)+v
    return out
def topk_with_prefix(c: Dict[str,int], prefix: str, k: int=5) -> List[Tuple[str,int]]:
    cand=[(k,v) for k,v in c.items() if k.startswith(prefix)]
    cand.sort(key=lambda kv:(-kv[1],kv[0]))
    return cand[:k]
def pipe(texts: List[str], n: int=2) -> Dict[str,int]:
    blocks=[]
    for t in texts:
        w=split_words(t); g=build_ngrams(w,n); blocks.append(tally(g))
    return reduce_sum(blocks)
