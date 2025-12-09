"""
Bug Detection Module for LHDiff (Clean & Fixed Version)
Detects bug-introducing changes using git history
"""

import os
import re
import subprocess
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

# -------------------------
# Data classes
# -------------------------
@dataclass
class Commit:
    hash: str
    message: str
    author: str
    date: str
    files_changed: List[str]

    def is_bug_fix(self) -> bool:
        """Check if commit message indicates a bug-fix"""
        message_lower = self.message.lower()
        bug_keywords = ['fix', 'bug', 'patch', 'resolve', 'correct', 'repair', 'defect', 'crash', 'exception', 'close']
        issue_patterns = [r'#\d+', r'fixes #\d+', r'closes #\d+', r'issue-\d+', r'bug-\d+']
        for kw in bug_keywords:
            if kw in message_lower:
                return True
        for pattern in issue_patterns:
            if re.search(pattern, message_lower):
                return True
        return False

@dataclass
class BugIntroducingChange:
    introducing_commit: str
    introducing_commit_msg: str
    introducing_date: str
    fix_commit: str
    fix_commit_msg: str
    fix_date: str
    file_path: str
    lines_changed: List[int]
    code_snippet: str

# -------------------------
# BugDetector
# -------------------------
class BugDetector:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"{repo_path} is not a git repository")

    # -------------------------
    # Git command runner
    # -------------------------
    def run_git_command(self, command: List[str]) -> str:
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.stdout.decode('utf-8', errors='ignore').strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(command)}")
            if e.stderr:
                print(f"Error: {e.stderr.decode('utf-8', errors='ignore')}")
            return ""

    # -------------------------
    # Get recent commits
    # -------------------------
    def get_commit_history(self, max_commits: int = 100) -> List[Commit]:
        output = self.run_git_command(['log', f'-{max_commits}', '--pretty=format:%H|%an|%ai|%s'])
        if not output:
            return []

        commits = []
        for line in output.split('\n'):
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
            commit_hash, author, date, message = parts
            files = self.get_changed_files(commit_hash)
            commits.append(Commit(commit_hash, message, author, date, files))
        return commits

    # -------------------------
    # Get changed files in commit
    # -------------------------
    def get_changed_files(self, commit_hash: str) -> List[str]:
        output = self.run_git_command(['diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash])
        return [f for f in output.split('\n') if f]

    # -------------------------
    # Get lines changed in a commit
    # -------------------------
    def get_changed_lines(self, commit_hash: str, file_path: str) -> List[Tuple[int, str]]:
        output = self.run_git_command(['show', commit_hash, '--', file_path, '--unified=0'])
        if not output:
            return []

        changed_lines = []
        current_line = None
        for line in output.split('\n'):
            if line.startswith('@@'):
                match = re.search(r'\+(\d+)', line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith('+') and not line.startswith('+++'):
                if current_line is not None:
                    changed_lines.append((current_line, line[1:]))
                    current_line += 1
            elif line.startswith('-') and not line.startswith('---'):
                if current_line is not None:
                    changed_lines.append((current_line, line[1:]))
            elif current_line is not None and not line.startswith('\\'):
                current_line += 1
        return changed_lines

    # -------------------------
    # Git blame to find introducing commit
    # -------------------------
    def git_blame_line(self, file_path: str, line_number: int, commit_hash: str) -> Optional[str]:
        parent = self.run_git_command(['rev-parse', f'{commit_hash}^'])
        if not parent:
            return None

        output = self.run_git_command([
        'blame',
        '-L', f'{line_number},{line_number}',  # line range first
        parent,                                # then the commit hash
        '--',
        file_path,                             
        '--porcelain'
    ])
        if not output:
            return None
        first_line = output.split('\n')[0]
        blame_commit = first_line.split()[0]
        return blame_commit

    # -------------------------
    # Get commit info by hash
    # -------------------------
    def get_commit_info(self, commit_hash: str) -> Optional[Dict]:
        output = self.run_git_command(['log', '-1', commit_hash, '--pretty=format:%H|%an|%ai|%s'])
        if not output:
            return None
        parts = output.split('|', 3)
        if len(parts) < 4:
            return None
        return {'hash': parts[0], 'author': parts[1], 'date': parts[2], 'message': parts[3]}

    # -------------------------
    # Analyze bug-fix commits
    # -------------------------
    def analyze_bug_fixes(self, commits: List[Commit]) -> List[BugIntroducingChange]:
        print("\nAnalyzing bug-fix commits...")
        bug_fixes = [c for c in commits if c.is_bug_fix()]
        print(f"Found {len(bug_fixes)} bug-fix commits out of {len(commits)} total")
        bug_introducing_changes = []

        for fix_commit in bug_fixes:
            print(f"\nAnalyzing fix: {fix_commit.hash[:8]} - {fix_commit.message[:50]}")
            for file_path in fix_commit.files_changed:
                if "bug_detection.py" in file_path:
                    continue

                full_file = self.repo_path / file_path
                if full_file.exists() and full_file.stat().st_size > 50000:
                    continue

                changed_lines = self.get_changed_lines(fix_commit.hash, file_path)
                if not changed_lines:
                    continue

                print(f"  File: {file_path} -> {len(changed_lines)} changed lines")

                for line_num, content in changed_lines[:5]:
                    introducing_commit_hash = self.git_blame_line(file_path, line_num, fix_commit.hash)
                    if not introducing_commit_hash or introducing_commit_hash == fix_commit.hash:
                        continue

                    info = self.get_commit_info(introducing_commit_hash)
                    if not info:
                        continue

                    bic = BugIntroducingChange(
                        introducing_commit=introducing_commit_hash[:8],
                        introducing_commit_msg=info['message'],
                        introducing_date=info['date'],
                        fix_commit=fix_commit.hash[:8],
                        fix_commit_msg=fix_commit.message,
                        fix_date=fix_commit.date,
                        file_path=file_path,
                        lines_changed=[line_num],
                        code_snippet=content[:100]
                    )
                    bug_introducing_changes.append(bic)
        return bug_introducing_changes

    # -------------------------
    # Generate human-readable report
    # -------------------------
    def generate_report(self, bug_introducing_changes: List[BugIntroducingChange]) -> str:
        report = ["="*80, "BUG-INTRODUCING CHANGES REPORT", "="*80]
        report.append(f"\nTotal bug-introducing changes found: {len(bug_introducing_changes)}\n")

        by_commit = {}
        for bic in bug_introducing_changes:
            by_commit.setdefault(bic.introducing_commit, []).append(bic)

        report.append(f"Unique commits that introduced bugs: {len(by_commit)}\n")
        report.append("-"*80)

        for idx, (commit, changes) in enumerate(by_commit.items(), 1):
            first = changes[0]
            report.append(f"\n{idx}. BUG-INTRODUCING COMMIT: {first.introducing_commit}")
            report.append(f"   Message: {first.introducing_commit_msg}")
            report.append(f"   Date: {first.introducing_date}")
            report.append(f"   Bugs introduced: {len(changes)}\n")

            fixes = {}
            for change in changes:
                fixes.setdefault(change.fix_commit, []).append(change)

            report.append("   Fixed by:")
            for fix_commit, fix_changes in fixes.items():
                fc = fix_changes[0]
                report.append(f"   - {fix_commit}: {fc.fix_commit_msg}")
                report.append(f"     Date: {fc.fix_date}")
                report.append(f"     Files: {', '.join(set(c.file_path for c in fix_changes))}\n")
        report.append("="*80)
        return "\n".join(report)

    # -------------------------
    # Save JSON & Text reports
    # -------------------------
    def save_results(self, bug_introducing_changes: List[BugIntroducingChange], output_dir: str = "bonus"):
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'bug_introducing_changes.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([{
                'introducing_commit': bic.introducing_commit,
                'introducing_message': bic.introducing_commit_msg,
                'introducing_date': bic.introducing_date,
                'fix_commit': bic.fix_commit,
                'fix_message': bic.fix_commit_msg,
                'fix_date': bic.fix_date,
                'file': bic.file_path,
                'lines': bic.lines_changed,
                'code': bic.code_snippet
            } for bic in bug_introducing_changes], f, indent=2)
        print(f"\n✅ JSON results saved to: {json_path}")

        report_path = os.path.join(output_dir, 'bug_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report(bug_introducing_changes))
        print(f"✅ Text report saved to: {report_path}")

        print(self.generate_report(bug_introducing_changes))

    # -------------------------
    # Run full analysis
    # -------------------------
    def run(self, max_commits: int = 100):
        print("🐛 Starting Bug Detection Analysis")
        print("="*80)
        commits = self.get_commit_history(max_commits)
        if not commits:
            print("❌ No commits found!")
            return

        print(f"✅ Loaded {len(commits)} commits")
        bug_introducing_changes = self.analyze_bug_fixes(commits)

        if not bug_introducing_changes:
            print("\n⚠️  No bug-introducing changes detected")
            return

        self.save_results(bug_introducing_changes)
        print(f"\n🎉 Analysis complete! Found {len(bug_introducing_changes)} bug-introducing changes")

# -------------------------
# Main entry
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bug Detection using Git History")
    ap.add_argument("--repo", default=".", help="Path to the git repository")
    ap.add_argument("--max-commits", type=int, default=100, help="Maximum number of commits to analyze")
    args = ap.parse_args()

    detector = BugDetector(args.repo)
    detector.run(max_commits=args.max_commits)

if __name__ == "__main__":
    main()
