import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def calibration(probs, labels, n_bins =10):
    bin_bound = np.linspace(0,1, n_bins + 1)
    cal = 0

    for i in range(n_bins):
        bin_find = (probs > bin_bound[i]) & (probs <= bin_bound[i+1])
        if np.any(bin_find):
            acc = np.mean(labels[bin_find])
            conf = np.mean(probs[bin_find])

            cal =+np.abs(acc - conf) * (np.sum(bin_find)/ len(probs))
    
    return cal



    