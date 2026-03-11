# leaderboard/update_leaderboard.py

from pathlib import Path
import pandas as pd
import subprocess
import json
import sys
import time

# Resolve repo root
repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from encryption.decrypt import decrypt_file

# Submissions folder and leaderboard CSV
SUBMISSIONS_DIR = repo_root / "submissions"
LEADERBOARD_CSV = repo_root / "leaderboard/leaderboard.csv"

def ensure_metadata(team_dir, submission_type="ideal"):
    """Create metadata.json in team directory if missing."""
    metadata_file = team_dir / "metadata.json"
    
    # Only create if it doesn't exist
    if not metadata_file.exists():
        print(f"DEBUG: Creating metadata.json for {team_dir.name}")
        metadata = {
            "team_name": team_dir.name,
            "submission_time": "2026-03-11T20:00:00Z",
            "description": f"Auto-generated metadata for {team_dir.name}",
            "submission_type": submission_type
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"DEBUG: metadata.json created at {metadata_file}")
        
        # Verify it was created and is valid JSON
        if metadata_file.exists():
            print(f"DEBUG: metadata.json exists and size: {metadata_file.stat().st_size} bytes")
            # Verify it's valid JSON
            with open(metadata_file, 'r') as f:
                json.load(f)
            print(f"DEBUG: metadata.json contains valid JSON")
    else:
        print(f"DEBUG: metadata.json already exists for {team_dir.name}")
    
    return metadata_file

def get_leaderboard_data():
    leaderboard = []

    print(f"DEBUG: Repo root: {repo_root}")
    print(f"DEBUG: Looking for submissions in: {SUBMISSIONS_DIR}")
    
    if not SUBMISSIONS_DIR.exists():
        print("DEBUG: Submissions directory does not exist!")
        return leaderboard
    
    team_folders = [d for d in SUBMISSIONS_DIR.iterdir() if d.is_dir()]
    print("DEBUG: Found team folders:", [d.name for d in team_folders])

    for team_dir in team_folders:
        print(f"\nDEBUG: Processing team folder: {team_dir.name}")
        print("DEBUG: Files in team folder:", [f.name for f in team_dir.iterdir()])

        ideal_enc = team_dir / "ideal.enc"
        pert_enc = team_dir / "perturbed.enc"

        if not ideal_enc.exists() or not pert_enc.exists():
            print(f"Skipping {team_dir.name}: missing files (expected ideal.enc and perturbed.enc)")
            continue

        # Create metadata FIRST, before any decryption
        print(f"DEBUG: Ensuring metadata exists for {team_dir.name}")
        ensure_metadata(team_dir)

        # Decrypted CSV files
        ideal_csv = team_dir / "ideal_submissions.csv"
        pert_csv = team_dir / "perturbed_submission.csv"

        # Decrypt
        print(f"DEBUG: Decrypting {ideal_enc} -> {ideal_csv}")
        decrypt_file(ideal_enc, ideal_csv)
        print(f"DEBUG: Decrypting {pert_enc} -> {pert_csv}")
        decrypt_file(pert_enc, pert_csv)

        # Verify files exist after decryption
        print(f"DEBUG: After decryption - Files in team folder:", [f.name for f in team_dir.iterdir()])

        # Small delay to ensure file system is synced
        time.sleep(1)

        # Score ideal
        try:
            print(f"DEBUG: Scoring ideal submission: {ideal_csv}")
            print(f"DEBUG: Checking metadata.json exists: {(team_dir / 'metadata.json').exists()}")
            
            result = subprocess.run([
                sys.executable,
                str(repo_root / "leaderboard/score_submission.py"),
                str(ideal_csv),
                "--require-metadata"
            ], capture_output=True, text=True, check=True)
            
            print(f"DEBUG: Ideal scoring stdout: {result.stdout}")
            print(f"DEBUG: Ideal scoring stderr: {result.stderr}")
            
            ideal_scores = json.loads(result.stdout)
            print(f"DEBUG: Ideal scores parsed: {ideal_scores}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {ideal_csv}:")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            continue
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from ideal scoring: {e}")
            print(f"Output was: {result.stdout}")
            continue

        # Score perturbed
        try:
            print(f"DEBUG: Scoring perturbed submission: {pert_csv}")
            
            result = subprocess.run([
                sys.executable,
                str(repo_root / "leaderboard/score_submission.py"),
                str(pert_csv),
                "--require-metadata"
            ], capture_output=True, text=True, check=True)
            
            print(f"DEBUG: Perturbed scoring stdout: {result.stdout}")
            print(f"DEBUG: Perturbed scoring stderr: {result.stderr}")
            
            pert_scores = json.loads(result.stdout)
            print(f"DEBUG: Perturbed scores parsed: {pert_scores}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {pert_csv}:")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            continue
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from perturbed scoring: {e}")
            print(f"Output was: {result.stdout}")
            continue

        leaderboard.append({
            "team_name": team_dir.name,
            "validation_f1_ideal": ideal_scores.get("validation_f1_score", 0),
            "validation_f1_perturbed": pert_scores.get("validation_f1_score", 0),
            "robustness_gap": ideal_scores.get("validation_f1_score", 0) - pert_scores.get("validation_f1_score", 0)
        })

    return leaderboard

def update_leaderboard_csv():
    leaderboard_data = get_leaderboard_data()
    if not leaderboard_data:
        print("No submissions found")
        return

    df = pd.DataFrame(leaderboard_data)
    df = df.sort_values(
        ["validation_f1_perturbed", "robustness_gap"], ascending=[False, True]
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(LEADERBOARD_CSV, index=False)
    print(f"Updated leaderboard at {LEADERBOARD_CSV}")
    print("DEBUG: Leaderboard data:")
    print(df.to_dict(orient="records"))

if __name__ == "__main__":
    update_leaderboard_csv()
