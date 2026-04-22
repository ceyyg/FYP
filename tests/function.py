import numpy as np
import matplotlib.pyplot as plt
from src.trial import fix_age, collapse_age
from src.metrics import ece_score, compute_all_metrics 
def run_unit_tests():
    print("UNIT TESTING:\n")

    # fix_age() test
    test_ages = [('oct-19', '10-19'), ('03-sep', '3-9'), ('20-29', '20-29')]
    for inp, exp in test_ages:
        res = fix_age(inp)
        print(f"[fix_age] Input: '{inp}'- Expected: '{exp}' - Result: '{res}'")
        assert res == exp
    print("fix_age() logic verified.\n")

    # collapse_age() test
    test_groups = [('0-2', 'Young'), ('30-39', 'Middle'), ('more than 70', 'Old')]
    for inp, exp in test_groups:
        res = collapse_age(inp)
        print(f"[collapse_age] Input: '{inp}' - Expected: '{exp}' - Result: '{res}'")
        assert res == exp
    print("collapse_age() logic verified.\n")

    # ece_score() logic test
    # Testing boundaries: Perfect confidence vs Total Miscalibration
    perf_probs, perf_labels = np.array([0.0, 1.0]), np.array([0, 1])
    bad_probs, bad_labels = np.array([1.0, 1.0]), np.array([0, 0])
    
    ece_perf = ece_score(perf_probs, perf_labels)
    ece_bad = ece_score(bad_probs, bad_labels)
    
    print(f"[ece_score] Perfect Calibration Result: {ece_perf:.4f}")
    print(f"[ece_score] Maximum Miscalibration Result: {ece_bad:.4f}")
    assert np.isclose(ece_perf, 0.0) and np.isclose(ece_bad, 1.0)
    print("ece_score() boundary tests passed.\n")

    # compute_all_metrics() logic test
    print("[compute_all_metrics] Simulating 300-sample stratified dataset...")
    # Creating controlled groups: A(100% acc), B(80% acc), C(60% acc)
    all_labels = np.array([1]*50 + [0]*50 + [1]*50 + [0]*50 + [1]*50 + [0]*50)
    all_preds = np.array([1]*50 + [0]*50 + [1]*40 + [0]*10 + [0]*40 + [1]*10 + [1]*30 + [0]*20 + [0]*30 + [1]*20)
    all_races = ["A"]*100 + ["B"]*100 + ["C"]*100
    all_probs = np.array([0.9]*300)
    all_ages = ["Middle"]*300

    metrics, stats, _ = compute_all_metrics(all_preds, all_labels, all_races, all_probs, all_ages)
    
    print(f"Worst Accuracy Detected: {metrics['Worst_Accuracy']:.2f} (Group: {metrics['Worst_Subgroup']})")
    print(f"Calculated Accuracy Gap: {metrics['Accuracy_Gap']:.2f}")
    assert metrics['Worst_Accuracy'] == 0.6
    print("Fairness metric calculation logic verified.\n")

def audit_dataset(dataset, num_samples=5):
    plt.figure(figsize=(20, 10))

    # Select random indices across the dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # ImageNet normalization constants for reversal
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(indices):
        # Access data via the Dataset __getitem__
        image, label, race, age_group = dataset[idx]

        # Access data via the Dataframe .iloc
        raw_row = dataset.df.iloc[idx]
        df_race = raw_row['race']
        df_gender = raw_row['gender']
        df_age_raw = raw_row['age']
        df_filename = raw_row['file']

        # Prepare Image for Display
        img_display = image.numpy().transpose((1, 2, 0))
        img_display = np.clip(img_display * std + mean, 0, 1)

        # Visualization and Title Construction
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_display)

        # Display both the DF source and the Mapped output to ensure consistency
        title = (
            f"Index: {idx}\n"
            f"File: {df_filename.split('/')[-1]}\n"
            f"Race: {df_race}\n"
            f"Gender: {df_gender}\n"
            f"Age (Raw): {df_age_raw}\n"
            f"Age (Mapped): {age_group}"
        )

        plt.title(title, fontsize=10, loc='left', fontweight='bold')
        plt.axis('off')

    plt.suptitle("Unit Test: Final Intersectional Label & Index Verification", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_unit_tests()