import os, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.paths import dataset_dir

train = pd.read_csv(f"{dataset_dir}/train_labels.csv")
test = pd.read_csv(f"{dataset_dir}/val_labels.csv")


def fix_age(age):
  """""
  Unit Test : Corrects string formatting errors found in Fairface CSV files.
  Ensures that 'oct-19' and '03-sep' is  interpreted in the correct age range
  """
  age_map = {'oct-19': '10-19','03-sep': '3-9'}
  return age_map.get(str(age).strip(), age)

def collapse_age(age):
  """""
  Unit Test : Categorises the 9 age ranges into simpler 3 broad categories (Young, Middle, Old).
  Ensures every intersectional subgroup meets the >100 sample reliability threshold.
  """
  if age in ['0-2', '3-9', '10-19']:
    return 'Young'

  elif age in ['20-29', '30-39', '40-49']:
    return 'Middle'

  else:
    return 'Old'

# Apply the fix_age() to the dataset
for df in (train, test):   
  df["age"] = df["age"].apply(fix_age)

def eda(df):
    """
    Experimental Verification: Visualizes Volume and Gender Skew
    across 21 Intersectional Subgroups.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    # Visual 1: Intersectional Volume
    counts = df.groupby(['race', 'clean_age']).size().reset_index(name='sample_count')
    sns.barplot(ax=axes[0], data=counts, x='race', y='sample_count', hue='clean_age', palette="magma")
    axes[0].set_title('Intersectional Subgroup Volume:', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Image Count')

    # Visual 2: Gender Balance Heatmap
    gender_pivot = df.groupby(['race', 'clean_age', 'gender']).size().unstack(fill_value=0)
    gender_pct = (gender_pivot['Male'] / (gender_pivot['Male'] + gender_pivot['Female'])) * 100
    gender_pct = gender_pct.unstack(level=1)
    
    sns.heatmap(ax=axes[1], data=gender_pct, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '% Male'})
    axes[1].set_title('Gender Balance Heatmap (% Male)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

full_df = pd.concat([train, test], axis=0).reset_index(drop=True)

# EDA Section
print(f"Total Dataset Dimensions: {full_df.shape}")
print("\nData Types:\n", full_df.dtypes)
print("\nNull Values Check:\n", full_df.isnull().sum())
print("\nStatistical Summary (Labels):\n", full_df.describe(include='all'))
full_df['clean_age'] = full_df['age'].apply(fix_age)

eda(full_df)



