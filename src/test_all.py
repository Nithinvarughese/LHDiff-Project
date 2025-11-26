#!/usr/bin/env python3
"""
Test Runner for LHDiff Project
This script tests all the different parts of our implementation
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# had to add these paths so python can find our modules
project_root = Path(__file__).parent.parent if "src" in str(Path(__file__).parent) else Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# importing our modules - try different import styles because paths can be tricky
try:
    from preprocessing.normalize import preprocess_file
    from simhash.simhash import generate_candidates, simhash, hamming_distance
    from matching.matcher import match_lines
    from split_detection.split_detection import split_detection
    from diff.difference_detector import run_mapping
except ImportError:
    from src.preprocessing.normalize import preprocess_file
    from src.simhash.simhash import generate_candidates, simhash, hamming_distance
    from src.matching.matcher import match_lines
    from src.split_detection.split_detection import split_detection
    from src.diff.difference_detector import run_mapping


def print_header(text: str):
    """just prints a nice looking header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_subheader(text: str):
    """prints subsection headers"""
    print(f"\n--- {text} ---")


def find_file_pairs(dataset_dir: str = "dataset") -> List[Tuple[str, str, str]]:
    """
    looks through dataset folder and finds all the v1/v2 pairs
    returns list of tuples: (base_name, v1_path, v2_path)
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: can't find dataset directory: {dataset_dir}")
        return []
    
    files = os.listdir(dataset_dir)
    v1_files = sorted([f for f in files if f.endswith("_v1.py")])
    
    pairs = []
    for v1_file in v1_files:
        base_name = v1_file.replace("_v1.py", "")
        v2_file = f"{base_name}_v2.py"
        
        v1_path = os.path.join(dataset_dir, v1_file)
        v2_path = os.path.join(dataset_dir, v2_file)
        
        if os.path.exists(v2_path):
            pairs.append((base_name, v1_path, v2_path))
        else:
            print(f"Warning: missing v2 file for {base_name}")
    
    return pairs


def test_preprocessing(v1_path: str, v2_path: str):
    """test the preprocessing step - normalizing lines"""
    print_subheader("Testing Preprocessing")
    
    try:
        v1_lines = preprocess_file(v1_path)
        v2_lines = preprocess_file(v2_path)
        
        print(f"OK - V1 preprocessed: {len(v1_lines)} lines")
        print(f"OK - V2 preprocessed: {len(v2_lines)} lines")
        
        # show a few lines so we can see what it looks like
        print("\nFirst few V1 lines:")
        for i, line in enumerate(v1_lines[:3], 1):
            display = line[:60] + '...' if len(line) > 60 else line
            print(f"  {i}: {display}")
        
        return v1_lines, v2_lines
    
    except Exception as e:
        print(f"ERROR in preprocessing: {e}")
        return None, None


def test_simhash(v1_lines: List[str], v2_lines: List[str], k: int = 15):
    """test simhash - generates candidates for matching"""
    print_subheader("Testing SimHash Candidate Generation")
    
    try:
        # compute simhash for first lines just to see it work
        if v1_lines and v2_lines:
            hash1 = simhash(v1_lines[0])
            hash2 = simhash(v2_lines[0])
            dist = hamming_distance(hash1, hash2)
            
            print(f"SimHash for V1[0]: {hash1}")
            print(f"SimHash for V2[0]: {hash2}")
            print(f"Hamming distance: {dist}")
        
        # now generate candidates for all lines
        candidates = generate_candidates(v1_lines, v2_lines, k=k)
        
        print(f"\nGenerated candidates for {len(candidates)} lines")
        print(f"Using top-{k} candidates per line")
        
        # show some examples
        print("\nExample candidates:")
        for i in sorted(candidates.keys())[:3]:
            # just show first 5 candidates to keep output readable
            cand_list = [c + 1 for c in candidates[i][:5]]
            print(f"  Line {i+1} -> candidates: {cand_list}")
        
        return candidates
    
    except Exception as e:
        print(f"ERROR in simhash: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_matching(v1_lines: List[str], v2_lines: List[str], candidates: Dict = None):
    """test the matching algorithm using content and context similarity"""
    print_subheader("Testing Line Matching")
    
    try:
        result = match_lines(
            v1_lines,
            v2_lines,
            candidates=candidates,
            w_content=0.6,
            w_context=0.4,
            threshold=0.55,
            radius=4
        )
        
        mappings = result.left_to_right
        
        print(f"Matched {len(mappings)} lines successfully")
        print(f"   Using threshold: 0.55")
        print(f"   Context radius: 4 lines")
        
        # show some of the mappings
        print("\nExample mappings (old -> new):")
        for old_idx in sorted(mappings.keys())[:10]:
            new_idx = mappings[old_idx]
            score = result.scores.get((old_idx, new_idx), 0.0)
            print(f"  {old_idx+1} -> {new_idx+1}  (similarity: {score:.3f})")
        
        return result
    
    except Exception as e:
        print(f"ERROR in matching: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_split_detection(v1_path: str, v2_path: str):
    """test split detection - finds lines that were split or merged"""
    print_subheader("Testing Split Detection")
    
    try:
        # capture output from split_detection since it prints directly
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        split_detection(v1_path, v2_path)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        print("Split detection completed successfully")
        
        # just show the last part of the output
        lines = output.strip().split('\n')
        print("\nSplit detection results (last 15 lines):")
        for line in lines[-15:]:
            print(f"  {line}")
        
        return True
    
    except Exception as e:
        sys.stdout = old_stdout
        print(f"ERROR in split detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diff_detector(v1_path: str, v2_path: str):
    """test the basic diff detector"""
    print_subheader("Testing Difference Detector")
    
    try:
        mapping = run_mapping(v1_path, v2_path)
        
        print(f"Generated {len(mapping)} mapping entries")
        
        # show first few mappings
        print("\nDiff detector results (first 10):")
        for m in mapping[:10]:
            print(f"  {m}")
        
        return mapping
    
    except Exception as e:
        print(f"ERROR in diff detector: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_pair(base_name: str, v1_path: str, v2_path: str):
    """runs all tests on a single file pair"""
    print_header(f"Testing: {base_name}")
    
    print(f"Old file: {v1_path}")
    print(f"New file: {v2_path}")
    
    # step 1: preprocessing
    v1_lines, v2_lines = test_preprocessing(v1_path, v2_path)
    if not v1_lines or not v2_lines:
        return False
    
    # step 2: simhash candidate generation
    candidates = test_simhash(v1_lines, v2_lines, k=15)
    
    # step 3: matching with candidates
    print_subheader("Matching WITH SimHash candidates")
    if candidates:
        result_with_candidates = test_matching(v1_lines, v2_lines, candidates)
    
    # step 4: matching without candidates (compare all pairs)
    print_subheader("Matching WITHOUT candidates (baseline)")
    result_without_candidates = test_matching(v1_lines, v2_lines, None)
    
    # step 5: split detection
    test_split_detection(v1_path, v2_path)
    
    # step 6: diff detector
    test_diff_detector(v1_path, v2_path)
    
    return True


def run_all_tests(dataset_dir: str = "dataset", limit: int = None):
    """runs tests on all file pairs in the dataset"""
    print_header("LHDiff Complete System Test")
    
    pairs = find_file_pairs(dataset_dir)
    
    if not pairs:
        print("ERROR: no file pairs found in dataset!")
        return
    
    print(f"\nFound {len(pairs)} file pairs to test")
    
    if limit:
        pairs = pairs[:limit]
        print(f"   Testing first {limit} pairs only")
    
    # run tests on each pair
    success_count = 0
    for i, (base_name, v1_path, v2_path) in enumerate(pairs, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{len(pairs)}")
        print('='*80)
        
        if test_single_pair(base_name, v1_path, v2_path):
            success_count += 1
    
    # print summary at the end
    print_header("TEST SUMMARY")
    print(f"Total pairs tested: {len(pairs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(pairs) - success_count}")
    print(f"Success rate: {success_count/len(pairs)*100:.1f}%")


def test_specific_file(file_base: str, dataset_dir: str = "dataset"):
    """test one specific file by name"""
    v1_path = os.path.join(dataset_dir, f"{file_base}_v1.py")
    v2_path = os.path.join(dataset_dir, f"{file_base}_v2.py")
    
    if not os.path.exists(v1_path):
        print(f"ERROR: file not found: {v1_path}")
        return False
    
    if not os.path.exists(v2_path):
        print(f"ERROR: file not found: {v2_path}")
        return False
    
    return test_single_pair(file_base, v1_path, v2_path)


def quick_test():
    """quick test on just file1"""
    print_header("QUICK TEST - just testing file1")
    test_specific_file("file1")


def main():
    """main function - handles command line args"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test all LHDiff components"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="test all file pairs"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="test specific file (eg file1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="limit number of files to test"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="quick test on file1 only"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="dataset directory (default: dataset)"
    )
    
    args = parser.parse_args()
    
    # run quick test
    if args.quick:
        quick_test()
        return
    
    # test one specific file
    if args.file:
        test_specific_file(args.file, args.dataset)
        return
    
    # test all files
    if args.all:
        run_all_tests(args.dataset, limit=args.limit)
        return
    
    # if no args, show usage
    print("LHDiff Test Runner")
    print("\nUsage:")
    print("  python test_all.py --quick              # quick test on file1")
    print("  python test_all.py --file file10        # test specific file")
    print("  python test_all.py --all                # test all files")
    print("  python test_all.py --all --limit 3      # test first 3 files")
    print("\nRun with -h for help")


if __name__ == "__main__":
    main()