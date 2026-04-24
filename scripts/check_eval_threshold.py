"""CI gate: fail the build if hallucination rate exceeds threshold.

Reads DeepEval results and exits with code 1 if hallucination is too high.
Used in GitHub Actions to block merges with degraded eval scores.
"""

import csv
import sys
import os

MAX_HALLUCINATION_RATE = 0.10  # 10% threshold
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation", "eval_results.csv")


def check_threshold():
    if not os.path.exists(RESULTS_PATH):
        print("No evaluation results found. Skipping threshold check.")
        sys.exit(0)

    with open(RESULTS_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No evaluation data. Skipping.")
        sys.exit(0)

    # Get the most recent run (all rows with the latest timestamp prefix)
    latest_timestamp = rows[-1]["timestamp"][:10]  # date portion
    recent_rows = [r for r in rows if r["timestamp"][:10] == latest_timestamp]

    # Calculate hallucination rate
    hallucination_scores = []
    for row in recent_rows:
        score = row.get("hallucination_score", "N/A")
        if score not in ("N/A", "error"):
            hallucination_scores.append(float(score))

    if not hallucination_scores:
        print("No valid hallucination scores found. Skipping.")
        sys.exit(0)

    avg_hallucination = sum(hallucination_scores) / len(hallucination_scores)
    # In DeepEval, lower hallucination score = better (0 = no hallucination)
    hallucination_rate = 1 - avg_hallucination

    print(f"Hallucination rate: {hallucination_rate:.2%}")
    print(f"Threshold: {MAX_HALLUCINATION_RATE:.2%}")
    print(f"Samples evaluated: {len(hallucination_scores)}")

    if hallucination_rate > MAX_HALLUCINATION_RATE:
        print(f"\nFAILED: Hallucination rate {hallucination_rate:.2%} exceeds {MAX_HALLUCINATION_RATE:.2%}")
        sys.exit(1)
    else:
        print(f"\nPASSED: Hallucination rate is within acceptable bounds.")
        sys.exit(0)


if __name__ == "__main__":
    check_threshold()
