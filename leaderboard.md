# Leaderboard Setup

This repo uses encrypted submissions and automatic scoring:

- **Submissions**: Place your prediction CSV in the `submissions/` folder. Only **encrypted** files (`.enc`) are accepted for scoring; raw `.csv` files are git-ignored and must not be pushed.
- **Authoritative data**: `leaderboard/leaderboard.csv`
- **Auto-generated Markdown**: this file (`leaderboard.md`)
- **Interactive UI**: e.g. `docs/leaderboard.html` or the published GitHub Pages site.

---

## For competitors: how to submit

1. **Prepare your submission**
   - Save your predictions as a `.csv` in the **`submissions/`** folder with columns `filename` and `prediction` (see `submissions/sample_submission.csv`).
   - `.csv` files in `submissions/` are git-ignored; they will not be committed.

2. **Encrypt your submission**
   - From the project root, go into `submissions/` and run the encryption script:
     - **Linux / macOS:** `cd submissions` then `python encrypt_submissions.py`
     - **Windows:** `cd submissions` then `python encrypt_submissions.py`
   - This creates a `.enc` file next to each `.csv` (e.g. `my_submission.csv.enc`).

3. **Push the encrypted file**
   - Commit and push the new `.enc` file(s) in `submissions/` (e.g. via a Pull Request or as required by the challenge).

4. **Automatic scoring**
   - On **push to main**, a GitHub Action runs that:
     - Finds the **latest** `.enc` file in `submissions/`
     - Decrypts it using the organisers’ private key (via `SUBMISSION_PRIVATE_KEY` secret)
     - Scores it against the hidden test labels (`TEST_LABELS_CSV` secret)
     - Updates `leaderboard/leaderboard.csv` and `leaderboard.md`

5. **View the leaderboard**
   - Open the published leaderboard (e.g. the link in the main README), or
   - Enable GitHub Pages with source `main` and folder `/docs` to serve the leaderboard UI.

---

## For organisers

- **Secrets**: Configure `SUBMISSION_PRIVATE_KEY` and `TEST_LABELS_CSV` in the repository secrets. Hidden test labels must not be committed.
- **Publishing**: To publish the leaderboard, enable GitHub Pages and set the source to the `main` branch and the `/docs` folder (or the path where the leaderboard HTML/CSV are served).
