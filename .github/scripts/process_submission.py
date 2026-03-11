# .github/scripts/process_submission.py
import subprocess
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

def run_script(script_path: Path):
    result = subprocess.run([sys.executable, str(script_path)])
    if result.returncode != 0:
        print(f"ERROR: Script {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("Decrypting submission...")
    run_script(repo_root / "encryption/decrypt.py")

    print("Scoring submission...")
    run_script(repo_root / "leaderboard/score_submission.py")

    print("Updating leaderboard...")
    run_script(repo_root / "leaderboard/update_leaderboard.py")

if __name__ == "__main__":
    main()
