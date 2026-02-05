import argparse
import os
import sys

import numpy as np

# Allow running from either project root (`python -m evaluation.my_evaluation`)
# or from the `evaluation/` directory (`python my_evaluation.py`).
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.pick_eval import (  # noqa: E402
    get_results_event_detection,
    get_results_phase_identification,
    get_results_onset_determination,
)


def evaluate_from_csv(task1_csv: str, task23_csv: str) -> dict:
    detection = get_results_event_detection(task1_csv)
    phase = get_results_phase_identification(task23_csv)
    onset = get_results_onset_determination(task23_csv)

    p_rmse = float(np.sqrt(np.mean(onset["P_onset_diff"] ** 2)))
    s_rmse = float(np.sqrt(np.mean(onset["S_onset_diff"] ** 2)))

    return {
        "event_auc": float(detection["auc"]),
        "phase_auc": float(phase["auc"]),
        "p_rmse_s": p_rmse,
        "s_rmse_s": s_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate phase-picking metrics from SeisBench CSV exports (task1 + task23)."
    )
    parser.add_argument("--task1_csv", required=True, help="CSV path for event detection (task1).")
    parser.add_argument("--task23_csv", required=True, help="CSV path for phase ID + onset (task23).")
    parser.add_argument("--name", default="model", help="Label for printing.")
    args = parser.parse_args()

    metrics = evaluate_from_csv(args.task1_csv, args.task23_csv)
    print(f"\n{args.name}")
    print("=" * len(args.name))
    print(f"Event detection AUC: {metrics['event_auc']:.4f}")
    print(f"Phase identification AUC: {metrics['phase_auc']:.4f}")
    print(f"P-onset RMSE (s): {metrics['p_rmse_s']:.4f}")
    print(f"S-onset RMSE (s): {metrics['s_rmse_s']:.4f}")


if __name__ == "__main__":
    main()
