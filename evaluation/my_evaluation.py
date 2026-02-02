from pathlib import Path
import os, sys
# Allow running from either project root (python -m evaluation.my_evaluation)
# or from the evaluation/ directory (python my_evaluation.py)
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.pick_eval import (
    get_results_event_detection,
    get_results_phase_identification,
    get_results_onset_determination
)
import numpy as np

def evaluate_model(model_name, task1_path, task23_path):
    """Evaluate a single model using friend's functions"""
    print(f"\n🎯 {model_name} Results:")
    print("=" * 50)

    # 1. Event Detectaon (earthquake vs noise)
    detection = get_results_event_detection(task1_path)
    print(f"📊 Event Detection AUC: {detection['auc']:.4f}")

    # 2. Phase Identification (P vs S)
    phase = get_results_phase_identification(task23_path)
    print(f"📊 Phase Identification AUC: {phase['auc']:.4f}")

    # 3. Onset Determination (timing)
    onset = get_results_onset_determination(task23_path)
    p_rmse = np.sqrt(np.mean(onset['P_onset_diff']**2))
    s_rmse = np.sqrt(np.mean(onset['S_onset_diff']**2))
    print(f"📊 P-wave RMSE: {p_rmse:.4f}s")
    print(f"📊 S-wave RMSE: {s_rmse:.4f}s")

    return {
        'detection_auc': detection['auc'],
        'phase_auc': phase['auc'],
        'p_rmse': p_rmse,
        's_rmse': s_rmse
    }

def main():
    """Main evaluation function"""

    # File paths for both models
    models = {
        "287k Bidirectional xLSTM": {
            "task1": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-07-28__21_57_56/evals/eval_ETHZ/test_task1.csv"
            ),
            "task23": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-07-28__21_57_56/evals/eval_ETHZ/test_task23.csv"
            )
        },
        "270k Unidirectional xLSTM": {
            "task1": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-07-28__22_00_54/evals/eval_ETHZ/test_task1.csv"
            ),
            "task23": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-07-28__22_00_54/evals/eval_ETHZ/test_task23.csv"
            )
        },
        # Added: mLSTM sequential pretraining (20% ETHZ fine-tune)
        "mLSTM sequential pretraining 20%": {
            "task1": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-10-07__10_24_49/evals/eval_ETHZ/test_task1.csv"
            ),
            "task23": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-10-07__10_24_49/evals/eval_ETHZ/test_task23.csv"
            )
        },

        "mLSTM sequential pretraining nounfreeze 20%": {
            "task1": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-10-09__00_17_57/evals/eval_ETHZ/test_task1.csv"
            ),
            "task23": (
                "/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/"
                "wandb_logs/mars/2025-10-09__00_17_57/evals/eval_ETHZ/test_task23.csv"
            )
        }
    }

    # Evaluate all models
    results = {}
    for model_name, paths in models.items():
        try:
            results[model_name] = evaluate_model(
                model_name,
                paths["task1"],
                paths["task23"]
            )
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("📈 SUMMARY COMPARISON")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n🔹 {model_name}:")
        print(f"   Detection AUC: {metrics['detection_auc']:.4f}")
        print(f"   Phase ID AUC: {metrics['phase_auc']:.4f}")
        print(f"   P-wave RMSE: {metrics['p_rmse']:.4f}s")
        print(f"   S-wave RMSE: {metrics['s_rmse']:.4f}s")

if __name__ == "__main__":
    main()
