from pathlib import Path

import pandas as pd

from .calculate_scores import get_leaderboard_data


def update_leaderboard_csv() -> None:
    leaderboard_data = get_leaderboard_data()
    output_path = Path(__file__).resolve().parent / "leaderboard.csv"
    df = pd.DataFrame(leaderboard_data)
    df.to_csv(output_path, index=False)
    print(f"Updated leaderboard at {output_path}")


if __name__ == "__main__":
    update_leaderboard_csv()
