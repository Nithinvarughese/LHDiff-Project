
"""
LHDiff - Complete Implementation
A language-independent hybrid line mapping technique

This combines all components:
1. Preprocessing
2. Unchanged line detection (Unix diff)
3. SimHash candidate generation
4. Content + Context similarity matching
5. Split detection
"""

import hashlib
import re
import argparse
from collections import Counter
from math import sqrt
from typing import List, Dict, Tuple, Optional
import difflib


# ============================================================================
# PREPROCESSING MODULE
# ============================================================================

def normalize_line(line: str) -> str:
    """Normalize a single line: lowercase and collapse whitespace"""
    line = line.strip().lower()
    return " ".join(line.split())


def preprocess_lines(lines: List[str]) -> List[str]:
    """Preprocess a list of lines"""
    return [normalize_line(line) for line in lines if line.strip()]


def read_and_preprocess(filepath: str) -> List[str]:
    """Read file and return preprocessed lines"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return preprocess_lines(f.readlines())


# ============================================================================
# SIMHASH MODULE
# ============================================================================

def simhash(text: str, hash_bits: int = 64) -> int:
    """Generate simhash fingerprint for text"""
    tokens = text.split()
    v = [0] * hash_bits

    for token in tokens:
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        for i in range(hash_bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if v[i] >= 0:
            fingerprint |= (1 << i)

    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculate hamming distance between two hashes"""
    x = hash1 ^ hash2
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist


def generate_candidates(old_lines: List[str], new_lines: List[str], k: int = 15) -> Dict[int, List[int]]:
    """Generate top-k candidates for each old line using simhash"""
    candidates = {}
    simhash_old = [simhash(line) for line in old_lines]
    simhash_new = [simhash(line) for line in new_lines]

    for i, old_hash in enumerate(simhash_old):
        distances = []
        for j, new_hash in enumerate(simhash_new):
            dist = hamming_distance(old_hash, new_hash)
            distances.append((j, dist))

        distances.sort(key=lambda x: x[1])
        candidates[i] = [idx for idx, _ in distances[:k]]

    return candidates


# ============================================================================
# SIMILARITY MODULE
# ============================================================================

def levenshtein_distance(a: str, b: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    
    if len(a) < len(b):
        a, b = b, a
    
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    
    return previous[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Calculate normalized Levenshtein similarity (0 to 1)"""
    if not a and not b:
        return 1.0
    
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (dist / max_len)


def get_context(lines: List[str], index: int, radius: int = 4) -> List[str]:
    """Get context lines around a given index"""
    lo = max(0, index - radius)
    hi = min(len(lines), index + radius + 1)
    return lines[lo:index] + lines[index + 1:hi]


def bow_vector(lines: List[str]) -> Counter:
    """Create bag-of-words vector from lines"""
    counter = Counter()
    for line in lines:
        for token in line.split():
            counter[token] += 1
    return counter


def cosine_similarity(v1: Counter, v2: Counter) -> float:
    """Calculate cosine similarity between two bow vectors"""
    if not v1 or not v2:
        return 0.0
    
    dot = sum(v1[term] * v2.get(term, 0) for term in v1)
    
    if dot == 0.0:
        return 0.0
    
    norm1 = sqrt(sum(c * c for c in v1.values()))
    norm2 = sqrt(sum(c * c for c in v2.values()))
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    return dot / (norm1 * norm2)


def context_similarity(left_ctx: List[str], right_ctx: List[str]) -> float:
    """Calculate context similarity using cosine similarity"""
    return cosine_similarity(bow_vector(left_ctx), bow_vector(right_ctx))


# ============================================================================
# UNCHANGED LINE DETECTION (Step 2)
# ============================================================================

def detect_unchanged_lines(old_lines: List[str], new_lines: List[str]) -> Tuple[Dict[int, int], set, set]:
    """
    Use difflib to find unchanged lines
    Returns: (mapping dict, used_old indices, used_new indices)
    """
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    mapping = {}
    used_old = set()
    used_new = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for old_idx, new_idx in zip(range(i1, i2), range(j1, j2)):
                mapping[old_idx] = new_idx
                used_old.add(old_idx)
                used_new.add(new_idx)

    return mapping, used_old, used_new


# ============================================================================
# MATCHING MODULE (Step 3 & 4)
# ============================================================================

class LHDiffMatcher:
    """Main matching class that implements LHDiff algorithm"""
    
    def __init__(
        self,
        old_lines: List[str],
        new_lines: List[str],
        w_content: float = 0.6,
        w_context: float = 0.4,
        threshold: float = 0.55,
        radius: int = 4,
        k: int = 15
    ):
        self.old_lines = old_lines
        self.new_lines = new_lines
        self.w_content = w_content
        self.w_context = w_context
        self.threshold = threshold
        self.radius = radius
        self.k = k
        
        # Pre-compute contexts
        self.old_contexts = [get_context(old_lines, i, radius) for i in range(len(old_lines))]
        self.new_contexts = [get_context(new_lines, i, radius) for i in range(len(new_lines))]
        
        # Cache for similarity scores
        self.score_cache = {}
    
    def compute_similarity(self, old_idx: int, new_idx: int) -> float:
        """Compute combined similarity score for a pair of lines"""
        key = (old_idx, new_idx)
        if key in self.score_cache:
            return self.score_cache[key]
        
        # Content similarity
        content_sim = normalized_levenshtein(
            self.old_lines[old_idx],
            self.new_lines[new_idx]
        )
        
        # Context similarity
        context_sim = context_similarity(
            self.old_contexts[old_idx],
            self.new_contexts[new_idx]
        )
        
        # Combined score
        score = self.w_content * content_sim + self.w_context * context_sim
        self.score_cache[key] = score
        
        return score
    
    def match_with_candidates(
        self,
        candidates: Dict[int, List[int]],
        used_old: set,
        used_new: set
    ) -> Dict[int, int]:
        """
        Step 4: Match lines using candidates and resolve conflicts
        """
        mapping = {}
        new_owner = {}  # Track which old line owns each new line
        
        for old_idx in range(len(self.old_lines)):
            if old_idx in used_old:
                continue
            
            if old_idx not in candidates or not candidates[old_idx]:
                continue
            
            best_new_idx = None
            best_score = self.threshold
            
            # Find best match among candidates
            for new_idx in candidates[old_idx]:
                if new_idx in used_new:
                    continue
                
                score = self.compute_similarity(old_idx, new_idx)
                
                if score > best_score:
                    best_score = score
                    best_new_idx = new_idx
            
            if best_new_idx is None:
                continue
            
            # Resolve conflicts (if new line already matched)
            if best_new_idx in new_owner:
                prev_old_idx = new_owner[best_new_idx]
                prev_score = self.compute_similarity(prev_old_idx, best_new_idx)
                
                if best_score > prev_score:
                    # New match is better, replace
                    del mapping[prev_old_idx]
                    mapping[old_idx] = best_new_idx
                    new_owner[best_new_idx] = old_idx
                # else: keep previous match
            else:
                # No conflict, add mapping
                mapping[old_idx] = best_new_idx
                new_owner[best_new_idx] = old_idx
        
        return mapping


# ============================================================================
# SPLIT DETECTION (Step 5)
# ============================================================================

def detect_splits(
    old_lines: List[str],
    new_lines: List[str],
    mapping: Dict[int, int],
    used_old: set,
    used_new: set
) -> Dict[int, List[int]]:
    """
    Detect line splits: one old line -> multiple new lines
    """
    split_mapping = {}
    
    for old_idx in range(len(old_lines)):
        if old_idx in used_old:
            continue
        
        old_clean = re.sub(r'\s+', '', old_lines[old_idx])
        
        # Try to find consecutive new lines that reconstruct the old line
        for start_idx in range(len(new_lines)):
            if start_idx in used_new:
                continue
            
            combined = ''
            new_indices = []
            
            for new_idx in range(start_idx, len(new_lines)):
                if new_idx in used_new:
                    break
                
                combined += re.sub(r'\s+', '', new_lines[new_idx])
                new_indices.append(new_idx)
                
                # Check if we've reconstructed the old line
                similarity = normalized_levenshtein(old_clean, combined)
                
                if similarity > 0.8 and len(new_indices) > 1:
                    split_mapping[old_idx] = new_indices
                    used_old.add(old_idx)
                    used_new.update(new_indices)
                    break
                
                # Stop if combined is much longer than old
                if len(combined) > len(old_clean) * 1.5:
                    break
            
            if old_idx in split_mapping:
                break
    
    return split_mapping


# ============================================================================
# MAIN LHDIFF ALGORITHM
# ============================================================================

def lhdiff(
    old_file: str,
    new_file: str,
    w_content: float = 0.6,
    w_context: float = 0.4,
    threshold: float = 0.55,
    radius: int = 4,
    k: int = 15,
    detect_splits_flag: bool = True
) -> Dict[int, List[int]]:
    """
    Complete LHDiff algorithm
    
    Returns: mapping from old line numbers to new line numbers (1-indexed)
             Each old line maps to a list of new lines (for splits)
    """
    
    # Step 1: Preprocessing
    old_lines = read_and_preprocess(old_file)
    new_lines = read_and_preprocess(new_file)
    
    # Step 2: Detect unchanged lines
    unchanged_mapping, used_old, used_new = detect_unchanged_lines(old_lines, new_lines)
    
    # Step 3: Generate candidates using SimHash
    candidates = generate_candidates(old_lines, new_lines, k=k)
    
    # Filter candidates to only unused lines
    for old_idx in candidates:
        if old_idx not in used_old:
            candidates[old_idx] = [j for j in candidates[old_idx] if j not in used_new]
    
    # Step 4: Resolve conflicts using content + context similarity
    matcher = LHDiffMatcher(old_lines, new_lines, w_content, w_context, threshold, radius, k)
    changed_mapping = matcher.match_with_candidates(candidates, used_old, used_new)
    
    # Merge unchanged and changed mappings
    final_mapping = {}
    for old_idx, new_idx in unchanged_mapping.items():
        final_mapping[old_idx] = [new_idx]
    
    for old_idx, new_idx in changed_mapping.items():
        final_mapping[old_idx] = [new_idx]
        used_old.add(old_idx)
        used_new.add(new_idx)
    
    # Step 5: Detect line splits (optional)
    if detect_splits_flag:
        split_mapping = detect_splits(old_lines, new_lines, final_mapping, used_old, used_new)
        final_mapping.update(split_mapping)
    
    # Convert to 1-indexed
    result = {}
    for old_idx, new_indices in final_mapping.items():
        result[old_idx + 1] = [n + 1 for n in new_indices]
    
    return result


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_mapping(mapping: Dict[int, List[int]]) -> List[str]:
    """Format mapping for output"""
    lines = []
    for old_idx in sorted(mapping.keys()):
        new_indices = mapping[old_idx]
        if len(new_indices) == 1:
            lines.append(f"{old_idx}->{new_indices[0]}")
        else:
            new_str = ",".join(str(n) for n in new_indices)
            lines.append(f"{old_idx}->{new_str}")
    return lines


def save_mapping(mapping: Dict[int, List[int]], output_file: str):
    """Save mapping to file"""
    lines = format_mapping(mapping)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def print_statistics(mapping: Dict[int, List[int]], old_file: str, new_file: str):
    """Print statistics about the mapping"""
    with open(old_file, 'r') as f:
        old_lines = [l for l in f if l.strip()]
    with open(new_file, 'r') as f:
        new_lines = [l for l in f if l.strip()]
    
    total_old = len(old_lines)
    total_new = len(new_lines)
    mapped = len(mapping)
    splits = sum(1 for v in mapping.values() if len(v) > 1)
    
    print(f"\n{'='*60}")
    print(f"LHDiff Statistics")
    print(f"{'='*60}")
    print(f"Old file lines:        {total_old}")
    print(f"New file lines:        {total_new}")
    print(f"Mapped lines:          {mapped}")
    print(f"Unmapped old lines:    {total_old - mapped}")
    print(f"Split lines detected:  {splits}")
    print(f"Mapping coverage:      {mapped/total_old*100:.1f}%")
    print(f"{'='*60}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LHDiff - Language-independent hybrid line mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python lhdiff.py --old file_v1.py --new file_v2.py
  
  # Save output to file
  python lhdiff.py --old file_v1.py --new file_v2.py --output mapping.txt
  
  # Adjust parameters
  python lhdiff.py --old file_v1.py --new file_v2.py --threshold 0.6 --k 20
  
  # Disable split detection
  python lhdiff.py --old file_v1.py --new file_v2.py --no-splits
        """
    )
    
    parser.add_argument('--old', required=True, help='Old version file path')
    parser.add_argument('--new', required=True, help='New version file path')
    parser.add_argument('--output', '-o', help='Output file for mapping')
    parser.add_argument('--threshold', type=float, default=0.55,
                       help='Similarity threshold (default: 0.55)')
    parser.add_argument('--w-content', type=float, default=0.6,
                       help='Weight for content similarity (default: 0.6)')
    parser.add_argument('--w-context', type=float, default=0.4,
                       help='Weight for context similarity (default: 0.4)')
    parser.add_argument('--radius', type=int, default=4,
                       help='Context radius in lines (default: 4)')
    parser.add_argument('--k', type=int, default=15,
                       help='Number of candidates per line (default: 15)')
    parser.add_argument('--no-splits', action='store_true',
                       help='Disable split detection')
    parser.add_argument('--stats', action='store_true',
                       help='Print detailed statistics')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.w_content < 0 or args.w_context < 0:
        parser.error("Weights must be non-negative")
    if args.w_content + args.w_context == 0:
        parser.error("At least one weight must be > 0")
    
    print(f"\nRunning LHDiff...")
    print(f"Old file: {args.old}")
    print(f"New file: {args.new}")
    
    # Run LHDiff
    try:
        mapping = lhdiff(
            args.old,
            args.new,
            w_content=args.w_content,
            w_context=args.w_context,
            threshold=args.threshold,
            radius=args.radius,
            k=args.k,
            detect_splits_flag=not args.no_splits
        )
        
        # Print results
        if args.stats:
            print_statistics(mapping, args.old, args.new)
        
        lines = format_mapping(mapping)
        
        if args.output:
            save_mapping(mapping, args.output)
            print(f"✓ Mapping saved to: {args.output}")
        else:
            print(f"\nLine Mapping ({len(mapping)} matches):")
            print("="*60)
            for line in lines:
                print(line)
        
        print(f"\n✓ LHDiff completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())