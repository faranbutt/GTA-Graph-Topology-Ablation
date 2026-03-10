import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from encryption.encrypt import encrypt_file

def encrypt_submissions():
    SUBMISSION_DIR = "."
    files = [f for f in os.listdir(SUBMISSION_DIR) if f.endswith(".csv") and f != "sample_submission.csv"]

    for file in files:
        encrypt_file(file)


if __name__ == '__main__':
    encrypt_submissions()