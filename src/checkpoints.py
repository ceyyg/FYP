import pandas as pd
import os, json, torch
from paths import checkpts, gs_path
from data import device


def ckpt_path(tag, seed):
    """
    Generates a filename for each optimizer-seed combination to avoid overwriting.
    """
    return os.path.join(checkpts, f"{tag}_seed{seed}.pt")

def save_checkpoint(tag, seed, epoch, model, optimizer, best_val_loss, best_model_state, patience_count):
    """
    System Testing: Serialises the training to Drive.
    Stores weights, optimizer and early stopping counters.
    """
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "best_model_state": best_model_state,
        "patience_count": patience_count,
    }, ckpt_path(tag, seed))

def load_checkpoint(tag, model, optimizer, seed):
    """
    Restores the training state if a session is interrupted.
    """
    path = ckpt_path(tag, seed)

    # Starts from scratch if no echeckpoint exists
    if not os.path.exists(path):
        return 1, float("inf"), None, 0

    # Load state 
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    best_model_state = checkpoint["best_model_state"]
    patience_count = checkpoint["patience_count"]

    print(f"Resuming {tag} from epoch {start_epoch}")
    return start_epoch, best_val_loss, best_model_state, patience_count

def load_grid_search_state():
    """
    Loads the JSON dictionary of previously tested hyperparameters
    """
    if os.path.exists(gs_path):
        with open(gs_path, "r") as f:
            return json.load(f)
    return {}

def save_grid_search_state(state):
    """
    Saves the running hyperparameter tuning results to JSON file.
    """
    with open(gs_path, "w") as f:
        json.dump(state, f, indent=2)


def append_result_to_csv(metrics_row, result_path):
    """
    Writes the fairness metrics results to CSV in Drive.
    Implements duplication check to ensure the latest resulrs are recorded.
    """
    new_row_df = pd.DataFrame([metrics_row])

    if os.path.exists(result_path):
        old_df = pd.read_csv(result_path)
        updated_df = pd.concat([old_df, new_row_df], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=["optimizer", "seed"], keep="last")
    else:
        updated_df = new_row_df

    updated_df.to_csv(result_path, index=False)