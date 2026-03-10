import argparse
import json
from pathlib import Path

from calculate_scores import calculate_scores


def validate_metadata(submission_path: Path) -> None:
    metadata_path = submission_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json next to {submission_path.name}")
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata.json is invalid JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a single submission file.")
    parser.add_argument("submission_path", help="Path to predictions.csv")
    parser.add_argument("--require-metadata", action="store_true", help="Require metadata.json next to predictions.csv")
    args = parser.parse_args()

    submission_path = Path(args.submission_path).resolve()
    if args.require_metadata:
        validate_metadata(submission_path)

    scores = calculate_scores(submission_path)
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path

from calculate_scores import calculate_scores


def validate_metadata(submission_path: Path) -> None:
    metadata_path = submission_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json next to {submission_path.name}")
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata.json is invalid JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a single submission file.")
    parser.add_argument("submission_path", help="Path to predictions.csv")
    parser.add_argument("--require-metadata", action="store_true", help="Require metadata.json next to predictions.csv")
    args = parser.parse_args()

    submission_path = Path(args.submission_path).resolve()
    if args.require_metadata:
        validate_metadata(submission_path)

    scores = calculate_scores(submission_path)
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
