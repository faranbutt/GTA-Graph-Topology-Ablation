# leaderboard/update_leaderboard.py

from pathlib import Path
import pandas as pd
import subprocess
import json
import sys
import time
import os

# Resolve repo root
repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from encryption.decrypt import decrypt_file

# Submissions folder and leaderboard CSV (now points to docs/)
SUBMISSIONS_DIR = repo_root / "submissions"
LEADERBOARD_CSV = repo_root / "docs" / "leaderboard.csv"

def ensure_metadata(team_dir):
    """Create metadata.json in team directory if missing."""
    metadata_file = team_dir / "metadata.json"
    
    print(f"DEBUG: Checking metadata at: {metadata_file}")
    print(f"DEBUG: File exists? {metadata_file.exists()}")
    
    # Always recreate to be safe (for debugging)
    print(f"DEBUG: Creating/overwriting metadata.json for {team_dir.name}")
    metadata = {
        "team_name": team_dir.name,
        "submission_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": f"Auto-generated metadata for {team_dir.name}"
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"DEBUG: metadata.json created at {metadata_file}")
    
    # Verify it was created and is valid JSON
    if metadata_file.exists():
        print(f"DEBUG: metadata.json exists and size: {metadata_file.stat().st_size} bytes")
        # Verify it's valid JSON
        with open(metadata_file, 'r') as f:
            content = f.read()
            print(f"DEBUG: metadata.json content: {content}")
            json.loads(content)
        print(f"DEBUG: metadata.json contains valid JSON")
    else:
        print(f"DEBUG: ERROR - metadata.json was NOT created!")
    
    return metadata_file

def preview_csv_file(csv_path: Path, label="CSV"):
    """Preview a CSV file to see its columns and first few rows."""
    try:
        if csv_path.exists():
            print(f"\nDEBUG: Preview of {label} file: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"DEBUG: Columns: {list(df.columns)}")
            print(f"DEBUG: Shape: {df.shape}")
            print(f"DEBUG: First 3 rows:")
            print(df.head(3).to_string())
            print(f"DEBUG: Data types:\n{df.dtypes}")
            return df
        else:
            print(f"DEBUG: {label} file does not exist: {csv_path}")
            return None
    except Exception as e:
        print(f"DEBUG: Could not read {label} CSV: {e}")
        return None

def get_leaderboard_data():
    leaderboard = []

    print(f"DEBUG: Repo root: {repo_root}")
    print(f"DEBUG: Looking for submissions in: {SUBMISSIONS_DIR}")
    print(f"DEBUG: TEST_LABELS_CSV environment variable: {os.environ.get('TEST_LABELS_CSV', 'NOT SET')}")
    
    if not SUBMISSIONS_DIR.exists():
        print("DEBUG: Submissions directory does not exist!")
        return leaderboard
    
    team_folders = [d for d in SUBMISSIONS_DIR.iterdir() if d.is_dir()]
    print("DEBUG: Found team folders:", [d.name for d in team_folders])

    for team_dir in team_folders:
        print(f"\n{'='*50}")
        print(f"DEBUG: Processing team folder: {team_dir.name}")
        print(f"DEBUG: Absolute path: {team_dir.absolute()}")
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
        print("DEBUG: After decryption - Files in team folder:")
        for f in team_dir.iterdir():
            print(f"  {f.name} (size: {f.stat().st_size if f.exists() else 'N/A'})")

        # PREVIEW THE DECRYPTED CSV FILES
        ideal_df = preview_csv_file(ideal_csv, "Ideal Submission")
        pert_df = preview_csv_file(pert_csv, "Perturbed Submission")

        # Also preview the ground truth file if available
        truth_path = repo_root / "data" / "train.csv"
        preview_csv_file(truth_path, "Ground Truth (train.csv)")

        # Small delay to ensure file system is synced
        time.sleep(1)

        # Double-check metadata.json still exists
        metadata_file = team_dir / "metadata.json"
        print(f"DEBUG: Before scoring - metadata.json exists? {metadata_file.exists()}")
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                print(f"DEBUG: metadata.json content: {f.read()}")

        # Score ideal
        try:
            print(f"\nDEBUG: Scoring ideal submission: {ideal_csv}")
            print(f"DEBUG: Working directory: {os.getcwd()}")
            
            cmd = [
                sys.executable,
                str(repo_root / "leaderboard/score_submission.py"),
                str(ideal_csv.absolute()),
                "--require-metadata"
            ]
            print(f"DEBUG: Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(repo_root)
            )
            
            print(f"DEBUG: Ideal scoring stdout: {result.stdout}")
            if result.stderr:
                print(f"DEBUG: Ideal scoring stderr: {result.stderr}")
            
            ideal_scores = json.loads(result.stdout)
            print(f"DEBUG: Ideal scores parsed: {ideal_scores}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {ideal_csv}:")
            print(f"  return code: {e.returncode}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            continue
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from ideal scoring: {e}")
            if 'result' in locals():
                print(f"Output was: {result.stdout}")
            continue
        except Exception as e:
            print(f"Unexpected error scoring ideal: {e}")
            continue

        # Score perturbed
        try:
            print(f"\nDEBUG: Scoring perturbed submission: {pert_csv}")
            
            cmd = [
                sys.executable,
                str(repo_root / "leaderboard/score_submission.py"),
                str(pert_csv.absolute()),
                "--require-metadata"
            ]
            print(f"DEBUG: Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(repo_root)
            )
            
            print(f"DEBUG: Perturbed scoring stdout: {result.stdout}")
            if result.stderr:
                print(f"DEBUG: Perturbed scoring stderr: {result.stderr}")
            
            pert_scores = json.loads(result.stdout)
            print(f"DEBUG: Perturbed scores parsed: {pert_scores}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {pert_csv}:")
            print(f"  return code: {e.returncode}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            continue
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from perturbed scoring: {e}")
            if 'result' in locals():
                print(f"Output was: {result.stdout}")
            continue
        except Exception as e:
            print(f"Unexpected error scoring perturbed: {e}")
            continue

        leaderboard.append({
            "team_name": team_dir.name,
            "validation_f1_ideal": ideal_scores.get("validation_f1_score", 0),
            "validation_f1_perturbed": pert_scores.get("validation_f1_score", 0),
            "robustness_gap": ideal_scores.get("validation_f1_score", 0) - pert_scores.get("validation_f1_score", 0)
        })

    return leaderboard

def update_leaderboard_csv():
    print("DEBUG: Starting leaderboard update...")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Python executable: {sys.executable}")
    
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
