"""
Bug Detection Module for LHDiff
Identifies bug-introducing changes by analyzing bug-fix commits

Approach:
1. Scan commit history for bug-fix commits (keywords in messages)
2. Extract changed lines from bug-fix commits
3. Use git blame to find when those lines were introduced
4. Link bug-fixes to bug-introducing commits
"""

import os
import re
import subprocess
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Commit:
    """Represents a git commit"""
    hash: str
    message: str
    author: str
    date: str
    files_changed: List[str]
    
    def is_bug_fix(self) -> bool:
        """Check if this commit fixes a bug"""
        message_lower = self.message.lower()
        
        # Bug fix keywords
        bug_keywords = [
            'fix', 'bug', 'issue', 'error', 'patch',
            'correct', 'repair', 'resolve', 'closes',
            'defect', 'crash', 'exception'
        ]
        
        # Issue tracker patterns
        issue_patterns = [
            r'#\d+',           # GitHub: #123
            r'fixes #\d+',     # fixes #123
            r'closes #\d+',    # closes #123
            r'issue-\d+',      # issue-123
            r'bug-\d+',        # bug-123
        ]
        
        # Check keywords
        for keyword in bug_keywords:
            if keyword in message_lower:
                return True
        
        # Check issue patterns
        for pattern in issue_patterns:
            if re.search(pattern, message_lower):
                return True
        
        return False


@dataclass
class BugIntroducingChange:
    """Represents a bug-introducing change"""
    introducing_commit: str
    introducing_commit_msg: str
    introducing_date: str
    fix_commit: str
    fix_commit_msg: str
    fix_date: str
    file_path: str
    lines_changed: List[int]
    code_snippet: str


class BugDetector:
    """Detects bug-introducing changes using git history"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        
        # Check if it's a git repo
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"{repo_path} is not a git repository")
    
    def run_git_command(self, command: List[str]) -> str:
        """Run a git command and return output"""
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(command)}")
            print(f"Error: {e.stderr}")
            return ""
    
    def get_commit_history(self, max_commits: int = 100) -> List[Commit]:
        """Get recent commit history"""
        print(f"Fetching commit history (max {max_commits} commits)...")
        
        # Format: hash|author|date|message
        output = self.run_git_command([
            'log',
            f'-{max_commits}',
            '--pretty=format:%H|%an|%ai|%s'
        ])
        
        if not output:
            return []
        
        commits = []
        for line in output.split('\n'):
            if not line:
                continue
            
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
            
            commit_hash, author, date, message = parts
            
            # Get files changed in this commit
            files = self.get_changed_files(commit_hash)
            
            commits.append(Commit(
                hash=commit_hash,
                message=message,
                author=author,
                date=date,
                files_changed=files
            ))
        
        return commits
    
    def get_changed_files(self, commit_hash: str) -> List[str]:
        """Get list of files changed in a commit"""
        output = self.run_git_command([
            'diff-tree',
            '--no-commit-id',
            '--name-only',
            '-r',
            commit_hash
        ])
        
        return [f for f in output.split('\n') if f]
    
    def get_changed_lines(self, commit_hash: str, file_path: str) -> List[Tuple[int, str]]:
        """
        Get lines changed in a specific file for a commit
        Returns: [(line_number, content), ...]
        """
        # Get unified diff
        output = self.run_git_command([
            'show',
            f'{commit_hash}',
            '--',
            file_path,
            '--unified=0'  # No context lines
        ])
        
        if not output:
            return []
        
        changed_lines = []
        current_line = None
        
        for line in output.split('\n'):
            # Parse diff header: @@ -old +new @@
            if line.startswith('@@'):
                match = re.search(r'\+(\d+)', line)
                if match:
                    current_line = int(match.group(1))
            
            # Lines that were added (bug fixes)
            elif line.startswith('+') and not line.startswith('+++'):
                if current_line is not None:
                    content = line[1:]  # Remove '+'
                    changed_lines.append((current_line, content))
                    current_line += 1
            
            # Lines that were removed
            elif line.startswith('-') and not line.startswith('---'):
                # These were the buggy lines
                pass
            
            # Context lines
            elif current_line is not None and not line.startswith('\\'):
                current_line += 1
        
        return changed_lines
    
    def git_blame_line(self, file_path: str, line_number: int, commit_hash: str) -> Optional[str]:
        """
        Find which commit introduced a specific line
        Uses git blame on the parent of the fix commit
        """
        # Get parent commit
        parent = self.run_git_command(['rev-parse', f'{commit_hash}^'])
        if not parent:
            return None
        
        # Blame that line at parent commit
        output = self.run_git_command([
            'blame',
            '-L', f'{line_number},{line_number}',
            parent,
            '--',
            file_path,
            '--porcelain'
        ])
        
        if not output:
            return None
        
        # Extract commit hash from blame output
        first_line = output.split('\n')[0]
        blame_commit = first_line.split()[0]
        
        return blame_commit
    
    def analyze_bug_fixes(self, commits: List[Commit]) -> List[BugIntroducingChange]:
        """Analyze bug-fix commits to find bug-introducing changes"""
        print("\nAnalyzing bug-fix commits...")
        
        bug_fixes = [c for c in commits if c.is_bug_fix()]
        print(f"Found {len(bug_fixes)} bug-fix commits out of {len(commits)} total")
        
        bug_introducing_changes = []
        
        for fix_commit in bug_fixes:
            print(f"\nAnalyzing fix: {fix_commit.hash[:8]} - {fix_commit.message[:50]}")
            
            # Analyze each file changed in the fix
            for file_path in fix_commit.files_changed:
                # Skip non-Python files (optional)
                if not file_path.endswith('.py'):
                    continue
                
                print(f"  File: {file_path}")
                
                # Get lines changed in the fix
                changed_lines = self.get_changed_lines(fix_commit.hash, file_path)
                
                if not changed_lines:
                    continue
                
                print(f"    Changed {len(changed_lines)} lines")
                
                # For each changed line, find when it was introduced
                for line_num, content in changed_lines[:5]:  # Limit to first 5 lines
                    introducing_commit_hash = self.git_blame_line(
                        file_path, 
                        line_num, 
                        fix_commit.hash
                    )
                    
                    if not introducing_commit_hash:
                        continue
                    
                    # Get info about introducing commit
                    introducing_commit_info = self.get_commit_info(introducing_commit_hash)
                    
                    if not introducing_commit_info:
                        continue
                    
                    # Don't report if the fix is fixing its own commit
                    if introducing_commit_hash == fix_commit.hash:
                        continue
                    
                    # Create bug-introducing change record
                    bic = BugIntroducingChange(
                        introducing_commit=introducing_commit_hash[:8],
                        introducing_commit_msg=introducing_commit_info['message'],
                        introducing_date=introducing_commit_info['date'],
                        fix_commit=fix_commit.hash[:8],
                        fix_commit_msg=fix_commit.message,
                        fix_date=fix_commit.date,
                        file_path=file_path,
                        lines_changed=[line_num],
                        code_snippet=content[:100]  # First 100 chars
                    )
                    
                    bug_introducing_changes.append(bic)
                    
                    print(f"    Line {line_num}: introduced by {introducing_commit_hash[:8]}")
        
        return bug_introducing_changes
    
    def get_commit_info(self, commit_hash: str) -> Optional[Dict]:
        """Get commit information"""
        output = self.run_git_command([
            'log',
            '-1',
            commit_hash,
            '--pretty=format:%H|%an|%ai|%s'
        ])
        
        if not output:
            return None
        
        parts = output.split('|', 3)
        if len(parts) < 4:
            return None
        
        return {
            'hash': parts[0],
            'author': parts[1],
            'date': parts[2],
            'message': parts[3]
        }
    
    def generate_report(self, bug_introducing_changes: List[BugIntroducingChange]) -> str:
        """Generate a readable report"""
        report = []
        report.append("=" * 80)
        report.append("BUG-INTRODUCING CHANGES REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal bug-introducing changes found: {len(bug_introducing_changes)}\n")
        
        # Group by introducing commit
        by_commit = {}
        for bic in bug_introducing_changes:
            if bic.introducing_commit not in by_commit:
                by_commit[bic.introducing_commit] = []
            by_commit[bic.introducing_commit].append(bic)
        
        report.append(f"Unique commits that introduced bugs: {len(by_commit)}\n")
        report.append("-" * 80)
        
        # Report each bug-introducing commit
        for idx, (commit, changes) in enumerate(by_commit.items(), 1):
            first = changes[0]
            
            report.append(f"\n{idx}. BUG-INTRODUCING COMMIT: {first.introducing_commit}")
            report.append(f"   Message: {first.introducing_commit_msg}")
            report.append(f"   Date: {first.introducing_date}")
            report.append(f"   Bugs introduced: {len(changes)}")
            report.append("")
            
            # List fixes
            fixes = {}
            for change in changes:
                if change.fix_commit not in fixes:
                    fixes[change.fix_commit] = []
                fixes[change.fix_commit].append(change)
            
            report.append("   Fixed by:")
            for fix_commit, fix_changes in fixes.items():
                fc = fix_changes[0]
                report.append(f"   - {fix_commit}: {fc.fix_commit_msg}")
                report.append(f"     Date: {fc.fix_date}")
                report.append(f"     Files: {', '.join(set(c.file_path for c in fix_changes))}")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, bug_introducing_changes: List[BugIntroducingChange], output_dir: str = "bonus"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_data = []
        for bic in bug_introducing_changes:
            json_data.append({
                'introducing_commit': bic.introducing_commit,
                'introducing_message': bic.introducing_commit_msg,
                'introducing_date': bic.introducing_date,
                'fix_commit': bic.fix_commit,
                'fix_message': bic.fix_commit_msg,
                'fix_date': bic.fix_date,
                'file': bic.file_path,
                'lines': bic.lines_changed,
                'code': bic.code_snippet
            })
        
        json_path = os.path.join(output_dir, 'bug_introducing_changes.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✅ JSON results saved to: {json_path}")
        
        # Save as text report
        report = self.generate_report(bug_introducing_changes)
        report_path = os.path.join(output_dir, 'bug_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Text report saved to: {report_path}")
        
        # Print summary
        print(report)
    
    def run(self, max_commits: int = 100):
        """Run complete bug detection analysis"""
        print("🐛 Starting Bug Detection Analysis")
        print("=" * 80)
        
        # Get commit history
        commits = self.get_commit_history(max_commits)
        
        if not commits:
            print("❌ No commits found!")
            return
        
        print(f"✅ Loaded {len(commits)} commits")
        
        # Analyze bug fixes
        bug_introducing_changes = self.analyze_bug_fixes(commits)
        
        if not bug_introducing_changes:
            print("\n⚠️  No bug-introducing changes detected")
            print("   This could mean:")
            print("   - No bug-fix commits in history (good!)")
            print("   - Bug-fix commits don't reference earlier commits")
            print("   - Analysis needs more commits (try increasing max_commits)")
            return
        
        # Save results
        self.save_results(bug_introducing_changes)
        
        print(f"\n🎉 Analysis complete! Found {len(bug_introducing_changes)} bug-introducing changes")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect bug-introducing changes from git history"
    )
    parser.add_argument(
        '--repo',
        default='.',
        help='Path to git repository (default: current directory)'
    )
    parser.add_argument(
        '--max-commits',
        type=int,
        default=100,
        help='Maximum number of commits to analyze (default: 100)'
    )
    parser.add_argument(
        '--output',
        default='bonus',
        help='Output directory (default: bonus/)'
    )
    
    args = parser.parse_args()
    
    try:
        detector = BugDetector(args.repo)
        detector.run(args.max_commits)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nTo use bug detection:")
        print("1. Make sure you're in a git repository")
        print("2. Or specify a git repo: --repo /path/to/repo")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())