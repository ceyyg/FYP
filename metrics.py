import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from data import train, val, test, collapse_age

for df in (train, val, test):
  df['subgroup'] = df["race"] + "_" + df["age_group"]

print(train["subgroup"].nunique())
print(train["gender"].value_counts(normalize=True))
balance_table = train.groupby(["subgroup", "gender"]).size().unstack(fill_value=0)
balance_table["pct_male"] = balance_table.get("Male", 0) / balance_table.sum(axis=1)

print("Training subgroup balance:")
print(balance_table[["Male", "Female", "pct_male"]].sort_values("pct_male").to_string())


def ece_score(probs, labels, n_bins =10):
    """
    Unit Test: Measures model's confidence to predict
    Evaluates the gap between predicted confidence and observed accuracy.
    """
    bins = np.linspace(0,1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        if i == 0:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs > bins[i]) & (probs < bins[i + 1])

        if mask.any():
            confidence = probs[mask]. mean()
            accuracy = labels[mask].mean()
            ece += np.abs(accuracy - confidence) * (mask.sum() / len(probs))

    return ece

def compute_all_metrics(preds, labels, races, probs, ages):
    """
    Calculates bias across 21 race x age intersections.
    Produces accuracy gap and equalised odds metrics for evaluation.
    """
    overall_accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    overall_accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    f1_female = f1_score(labels, preds, pos_label=0)
    f1_male = f1_score(labels, preds, pos_label=1)

    precision_female = precision_score(labels, preds, pos_label=0)
    precision_male = precision_score(labels, preds, pos_label=1)

    recall_female = recall_score(labels, preds, pos_label=0)
    recall_male = recall_score(labels, preds, pos_label=1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metric_df = pd.DataFrame({
        "label": labels,
        "pred": preds,
        "race": races,
        "prob": probs,
        "age": ages
    })
    # Combine predictions and subgroup for grouping
    metric_df["subgroup"] = metric_df["race"] + "_" + metric_df["age"]

    subgroup_rows = []

    for subgroup in metric_df["subgroup"].unique():
        sub = metric_df[metric_df["subgroup"] == subgroup]
        tn, fp, fn, tp = confusion_matrix(sub["label"], sub["pred"], labels=[0, 1]).ravel()

        subgroup_rows.append({
            "subgroup": subgroup,
            "size": len(sub),
            "reliable": len(sub) >= 100,
            "Accuracy": accuracy_score(sub["label"], sub["pred"]),
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "ECE": ece_score(sub["prob"].values, sub["label"].values),
        })

    subgroup_stats = pd.DataFrame(subgroup_rows)
    reliable_stats = subgroup_stats[subgroup_stats["reliable"]]

    metrics = {
        "Overall_Accuracy": overall_accuracy,
        "Macro_F1": macro_f1,
        "F1_Female": f1_female,
        "F1_Male": f1_male,
        "Precision_Female": precision_female,
        "Precision_Male": precision_male,
        "Recall_Female": recall_female,
        "Recall_Male": recall_male,
        "Worst_Subgroup": reliable_stats.loc[reliable_stats["Accuracy"].idxmin(), "subgroup"],
        "Worst_Accuracy": reliable_stats["Accuracy"].min(),
        "Best_Accuracy": reliable_stats["Accuracy"].max(),
        "Accuracy_Gap": reliable_stats["Accuracy"].max() - reliable_stats["Accuracy"].min(),
        "EO_TPR_Range": reliable_stats["TPR"].max() - reliable_stats["TPR"].min(),
        "EO_TPR_Std": reliable_stats["TPR"].std(),
        "EO_FPR_Range": reliable_stats["FPR"].max() - reliable_stats["FPR"].min(),
        "EO_FPR_Std": reliable_stats["FPR"].std(),
        "Mean_ECE": reliable_stats["ECE"].mean(),
    }

    return metrics, subgroup_stats, cm