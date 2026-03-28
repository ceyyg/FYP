import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def calibration(probs, labels, n_bins =10):
    bin_bound = np.linspace(0,1, n_bins + 1)
    cali = 0

    for i in range(n_bins):
        bin_find = (probs > bin_bound[i]) & (probs <= bin_bound[i+1])
        if np.any(bin_find):
            acc = np.mean(labels[bin_find])
            conf = np.mean(probs[bin_find])

            cali =+np.abs(acc - conf) * (np.sum(bin_find)/ len(probs))
    
    return cali

def fairness_mets(all_preds, all_labels, all_races, all_probs):
    df = pd.DataFrame({
        'label': all_labels,
        'pred': all_preds,
        'race': all_races,
        'prob': all_probs
    })

    df['subgroup']= df['race'] + "_" + df['label'].map({1: 'Male', 0:'Female'})
    calcs = []
    for group in df['subgroup']. unique():
        sub = df[df['subgroup']== group]

        acc = accuracy_score(sub['label'], sub[pred])
        cali = calibration(sub['prob'].values, sub['label'.values])

        tn,fp,fn,tp = confusion_matrix(
            sub['label'], sub['pred'], labels=[0,1]).ravel()
        fpr = fp / (fp+tn) if (fp+tn) >0 else 0
        tpr = tp / (tp+fn) if (tp+fn) >0 else 0


        calcs.append({
            'subgroup': group,
            'Accuracy': acc,
            'FPR': fpr,
            'TPR': tpr,
            'Calibration': cali
        })

    stats_df = pd.DataFrame(calcs)

    metrics = {
        'Overall accuracy': accuracy_score(df['label', df['pred']]),
        'Worst Sub group': stats_df.loc[stats_df['Accuracy'].idxmin(), 'subgroup'],
        'Worst accuracy': stats_df['Accuracy'].min(),
        'Equal_odds_F': stats_df['FPR'].max() - stats_df['FPR'].min(),
        'Equal_odds_T': stats_df['TPR'].max() - stats_df['TPR'].min(),
        'Group calibration': stats_df['Calibration'].mean()
    }
    
    return metrics, stats_df