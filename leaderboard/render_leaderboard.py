from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "leaderboard.csv"
    md_path = base_dir / "leaderboard.md"
    docs_csv_path = base_dir.parent / "docs" / "leaderboard.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing leaderboard CSV at {csv_path}")

    df = pd.read_csv(csv_path)
    docs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(docs_csv_path, index=False)

    if df.empty:
        md_path.write_text("# Leaderboard\n\n_No submissions yet._\n", encoding="utf-8")
        return

    df = df.sort_values("validation_f1_score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))

    md = ["# Leaderboard", "", df.to_markdown(index=False)]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "leaderboard.csv"
    md_path = base_dir / "leaderboard.md"
    docs_csv_path = base_dir.parent / "docs" / "leaderboard.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing leaderboard CSV at {csv_path}")

    df = pd.read_csv(csv_path)
    docs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(docs_csv_path, index=False)
    if df.empty:
        md_path.write_text("# Leaderboard\n\n_No submissions yet._\n", encoding="utf-8")
        return

    df = df.sort_values("validation_f1_score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))

    md = ["# Leaderboard", "", df.to_markdown(index=False)]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
