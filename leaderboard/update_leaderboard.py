# scripts/leaderboard/update_leaderboard.py
from pathlib import Path
import pandas as pd
from .calculate_scores import calculate_scores_pair
from scripts.encryption.decrypt import decrypt_file

SUBMISSIONS_DIR = Path("submissions")
LEADERBOARD_CSV = Path(__file__).resolve().parent / "leaderboard.csv"


def get_leaderboard_data():
    leaderboard = []

    for team_dir in SUBMISSIONS_DIR.iterdir():
        if not team_dir.is_dir():
            continue

        ideal_enc = team_dir / "ideal.enc"
        pert_enc = team_dir / "perturbed.enc"

        if not ideal_enc.exists() or not pert_enc.exists():
            print(f"Skipping {team_dir.name}: missing files")
            continue

        # Decrypted paths
        ideal_csv = team_dir / "ideal.csv"
        pert_csv = team_dir / "perturbed.csv"

        # Decrypt
        decrypt_file(ideal_enc, ideal_csv)
        decrypt_file(pert_enc, pert_csv)

        # Calculate scores
        scores = calculate_scores_pair(ideal_csv, pert_csv)
        leaderboard.append({
            "team_name": team_dir.name,
            **scores
        })

    return leaderboard


def update_leaderboard_csv():
    leaderboard_data = get_leaderboard_data()
    if not leaderboard_data:
        print("No submissions found")
        return

    df = pd.DataFrame(leaderboard_data)

    # Rank: sort by perturbed F1 descending, then smaller gap wins
    df = df.sort_values(
        ["validation_f1_perturbed", "robustness_gap"],
        ascending=[False, True]
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    df.to_csv(LEADERBOARD_CSV, index=False)
    print(f"Updated leaderboard at {LEADERBOARD_CSV}")


if __name__ == "__main__":
    update_leaderboard_csv()
