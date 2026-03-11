from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from leaderboard.calculate_scores import calculate_scores_pair
from leaderboard.calculate_scores import calculate_scores

SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"


def get_participant_submissions():

    participants = {}

    for f in SUBMISSIONS_DIR.iterdir():

        if not f.is_file() or f.suffix != ".csv":
            continue

        name = f.stem.lower()

        # ---- Support ideal_submission.csv format ----
        if name == "ideal_submission":
            team = "participant"
            submission_type = "ideal"

        elif name == "perturbed_submission":
            team = "participant"
            submission_type = "perturbed"

        # ---- Support teamname_ideal.csv format ----
        elif "_" in name:
            team, submission_type = name.rsplit("_", 1)

            if submission_type not in ["ideal", "perturbed"]:
                continue
        else:
            continue

        participants.setdefault(team, {})[submission_type] = f

    return participants


def update_leaderboard_csv():

    participants = get_participant_submissions()

    rows = []

    for team, files in participants.items():

        if "ideal" not in files or "perturbed" not in files:
            print(f"Skipping {team} (missing ideal or perturbed submission)")
            continue

        print(f"Scoring submissions for team: {team}")

        ideal_scores = calculate_scores(files["ideal"])
        perturbed_scores = calculate_scores(files["perturbed"])

        ideal_f1 = ideal_scores["validation_f1_score"]
        perturbed_f1 = perturbed_scores["validation_f1_score"]

        row = {
            "team_name": team,
            "validation_f1_ideal": ideal_f1,
            "validation_f1_perturbed": perturbed_f1,
            "robustness_gap": ideal_f1 - perturbed_f1,
            "timestamp": datetime.fromtimestamp(
                files["ideal"].stat().st_mtime
            ).isoformat(),
        }

        rows.append(row)

    if not rows:
        print("No valid submissions found")
        return

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="validation_f1_ideal",
        ascending=False
    )

    output_path = Path(__file__).resolve().parent / "leaderboard.csv"

    df.to_csv(output_path, index=False)

    print(f"Leaderboard updated: {output_path}")


if __name__ == "__main__":
    update_leaderboard_csv()
