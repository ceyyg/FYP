import torch
import torch as nn
import numpy as np
import random, os, torch
import pandas as pd
from optimizer import build_optimizer, get_swats_phase
from data import device, data_loaders
from main import save_folder
from resnet import ResNet18
from checkpoints import load_checkpoint, save_checkpoint
from sklearn.metrics import accuracy_score
from metrics import compute_all_metrics


max_epoch = 10
patience = 3

def train_epoch(model, optimizer, loader, criterion):
    """
    Integration Test: Executes one full pass over the training set.
    Ensures weights are updated via backpropagation.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def validation(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    all_races = []
    all_probs = []
    all_ages = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    for images, labels, races, ages in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        all_races.extend(races)
        all_ages.extend(ages)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_races),
        np.array(all_probs),
        np.array(all_ages),
        acc,
        avg_loss
    )

def main_training(opt_name, lr, wd, seed, run_test=False):
    """
    System Framework: Manages the 10 epoch training and validation cycle.
    Includes checkpointing to resume training after interruption.
    """
    # Set the seed for this specific run
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize model, data, and optimizer
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(opt_name, model, lr, wd)
    train_loader, val_loader, test_loader = data_loaders(seed)

    tag = f"{opt_name}_lr{lr}_wd{wd}"  # Seed handled by checkpoint path function
    history_path = f"{save_folder}/{opt_name}_seed{seed}_history.csv"
    # Load checkpoint
    start_epoch, best_val_loss, best_model_state, patience_count = load_checkpoint(
        tag, model, optimizer, seed
    )

    # Safety net to ensure we have a state to save if starting fresh
    if best_model_state is None:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load old history if it exists
    if os.path.exists(history_path):
        history = pd.read_csv(history_path).to_dict("records")
    else:
        history = []

    swats_log = []

    print(f"\n Starting {opt_name} - Seed {seed} , LR {lr} , WD {wd} ")

    for epoch in range(start_epoch, max_epoch + 1):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, criterion)

        # Validation returns (preds, labels, races, probs, ages, acc, loss)
        _, _, _, _, _, val_acc, val_loss = validation(model, val_loader)

        # Log SWATS phase if applicable
        swats_phase = get_swats_phase(optimizer) if opt_name == "SWATS" else "N/A"
        swats_log.append((epoch, swats_phase))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "swats_phase": swats_phase
        })

        print(f"Epoch {epoch:02d} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Phase: {swats_phase}")

        # Improvement based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        # Save checkpoint every epoch
        save_checkpoint(
            tag=tag,
            seed=seed,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            best_model_state=best_model_state,
            patience_count=patience_count
        )

        # Save history every epoch
        pd.DataFrame(history).to_csv(history_path, index=False)

        effective_patience = max_epoch if opt_name == "SWATS" else patience
        if patience_count >= effective_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    result = {
        "best_val_acc": best_val_acc,
        "history": pd.DataFrame(history),
        "swats_log": swats_log,
        "swats_final_phase": swats_log[-1][1] if swats_log else "N/A",
        "swats_transition_epoch": next((ep for ep, phase in swats_log if phase == "SGD"), None)
    }

    # Performs the final system audit on the held-out test set
    if run_test:
        # Load the best performing version for the final audit
        model.load_state_dict(best_model_state)
        preds, labels, races, probs, ages, test_acc, test_loss = validation(model, test_loader)

        metrics, subgroup_stats, cm = compute_all_metrics(preds, labels, races, probs, ages)

        result.update({
            "test_metrics": metrics,
            "test_stats": subgroup_stats,
            "test_cm": cm,
            "test_acc": test_acc
        })
        print(f"Final Test Accuracy: {test_acc:.4f}")

    return result