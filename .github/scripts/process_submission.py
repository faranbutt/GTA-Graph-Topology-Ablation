# .github/scripts/process_submission.py
import subprocess
from pathlib import Path

# Repo root
repo_root = Path(__file__).parent.parent.parent.resolve()

def main():
    print("Decrypting submission...")
    subprocess.run(["python", str(repo_root / "encryption" / "decrypt.py")], check=True)

    print("Scoring submission...")
    subprocess.run(["python", str(repo_root / "leaderboard" / "score_submission.py")], check=True)

    print("Updating leaderboard...")
    subprocess.run(["python", str(repo_root / "leaderboard" / "update_leaderboard.py")], check=True)

if __name__ == "__main__":
    main()
