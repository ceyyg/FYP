import os, pandas as pd
import numpy as np
from gridsearch import best_hparams
from train import main_training
from trial import save_folder, result_path
from gridsearch import optimizers, eval_seeds
from checkpoints import append_result_to_csv


def run_optimizer(opt_name, eval_seeds):
    """
    Implementation of the runs of a specific optimizer across multiple seeds.
    """

    # Retriving the optimal configurations during the grid search
    lr = best_hparams[opt_name]["lr"]
    wd = best_hparams[opt_name]["weight_decay"]

    print(f"\nRunning: {opt_name}")
    print(f"Using lr={lr}, wd={wd}")

    for seed in eval_seeds:
        # Skip if the specific combination of optimizer and seed exists in the CSV
        if os.path.exists(result_path):
            existing_df = pd.read_csv(result_path)
            already_done = (
                (existing_df["optimizer"] == opt_name) &
                (existing_df["seed"] == seed)
            ).any()

            if already_done:
                print(f"Skipping {opt_name} seed {seed} (already in CSV)")
                continue

        # Run training on the held-out test set
        result = main_training(opt_name, lr, wd, seed, run_test=True)

        # Save final results
        metrics = result["test_metrics"].copy()
        metrics.update({
            "optimizer": opt_name,
            "seed": seed,
            "lr": lr,
            "wd": wd,
            "swats_final_phase": result["swats_final_phase"],
            "swats_transition_epoch": result["swats_transition_epoch"]
        })

        # Append the summary of the metrcis to CSV
        append_result_to_csv(metrics, result_path)

        # Save the stats into CSV files
        history_path = f"{save_folder}/{opt_name}_seed{seed}_history.csv"
        result["history"].to_csv(history_path, index=False)
        subgroup_path = f"{save_folder}/{opt_name}_seed{seed}_subgroups.csv"
        result["test_stats"].to_csv(subgroup_path, index=False)

        cm_path = f"{save_folder}/{opt_name}_seed{seed}_cm.npy"
        np.save(cm_path, result["test_cm"])

        # Log SWATS transition phase 
        if opt_name == "SWATS":
            swats_path = f"{save_folder}/{opt_name}_seed{seed}_swats_log.csv"
            pd.DataFrame(result["swats_log"], columns=["epoch", "phase"]).to_csv(swats_path, index=False)

        display_cols = ["subgroup", "size", "Accuracy", "TPR", "FPR", "ECE"]
        print(result["test_stats"][display_cols].sort_values("Accuracy").to_string(index=False))
        print(f"\n Saved everything for {opt_name}, seed {seed}")
        print(f"History: {history_path}")
        print(f"Subgroups: {subgroup_path}")
        print(f"Confusion Matrix: {cm_path}\n")


def exp_opt():
    """
    Coordinates the sequential processing of all optimizers.
    """
    print(f"Target Optimizers: {optimizers}")
    print(f"Evaluation seeds: {eval_seeds}")

    for opt_name in optimizers:
        run_optimizer(opt_name, eval_seeds)
        print("Experimentation completed.")

if __name__ == "__main__":
    exp_opt
