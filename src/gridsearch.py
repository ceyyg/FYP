import random
from checkpoints import load_grid_search_state, save_grid_search_state
from train import main_training

# Fixed seed for hyperparameter tuning
grid_seed = 42

# Defining learning rates and weight decays per optimizer for hypertuning
optimizers = ["SGD", "Adam", "SWATS"]
lr_configs = {
    "SGD":   [0.01, 0.001],
    "Adam":  [0.001, 0.0001],
    "SWATS": [0.001, 0.0001],
}

wd_configs = {
    "SGD":   [1e-3, 1e-4],
    "Adam":  [1e-3, 1e-4],
    "SWATS": [1e-3, 1e-4],
}

# Generates 3 random seeds 
random.seed(0)
eval_seeds = random.sample(range(1, 5000), 3)

print("Grid seed:", grid_seed)
print("Evaluation seeds:", eval_seeds)
print("Optimizers:", optimizers)

# Load previous state to allow resumption after interruption
grid_search_state = load_grid_search_state()
best_hparams = {}
all_grid_results = {}

for opt_name in optimizers:
    print(f"Grid search for {opt_name}")

    best_acc = -1
    best_lr = None
    best_wd = None

    # Loop to explore every configurations in the grid
    for lr in lr_configs[opt_name]:
        for wd in wd_configs[opt_name]:
            key = f"{opt_name}_lr{lr}_wd{wd}"

            # Load from saved JSON file if comboination was already tested
            if key in grid_search_state:
                val_acc = grid_search_state[key]
                print(f"[cached] {key} -> val_acc={val_acc:.4f}")
            else:
                # Perform 1 epoch training run to evaluate parameters
                result = main_training(opt_name, lr, wd, grid_seed, run_test=False)
                val_acc = result["best_val_acc"]
                grid_search_state[key] = float(val_acc)
                save_grid_search_state(grid_search_state)

            all_grid_results[key] = val_acc

            # Update best parameters based on Validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr
                best_wd = wd

    # Store the optimal settings 
    best_hparams[opt_name] = {
        "lr": best_lr,
        "weight_decay": best_wd,
        "best_val_acc": best_acc
    }

    print(f"Best for {opt_name}: lr={best_lr}, wd={best_wd}, val_acc={best_acc:.4f}")

print("\nSelected best hyperparameters:")
for opt_name, hp in best_hparams.items():
    print(opt_name, hp)