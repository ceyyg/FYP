# FairFace Fairness Evaluation: SGD vs. Adam vs. SWATS

## Project Overview
The core objective of the project is to analyze whether the choice of optimizer affects the predictive and fairness performance between different age, gender, and racial groups. The study specifically compares **SGD**, **Adam**, and **SWATS** (Switching from Adam to SGD) across 21 demographic subgroups using tha **Fairface** dataset.

### Key Features:
* **Grid Search:** Hyperparameter tuning for learning rate and weight decay.
* **SWATS Implementation:** Real-time monitoring of the switching criteria from Adam to SGD.
* **Robust Evaluation:** Multi-seed training (3 seeds) to ensure statistical significance.
* **Fairness Metrics:** Evaluated using calculation of Accuracy, TPR, FPR, and ECE per subgroup.

---
## Dataset
The project utilizes the FairFace dataset, a face image dataste designed to reduce bias in facial analysis algorithms available for academic purposes.
```bash
Source: https://github.com/dchen236/FairFace 
```

## Setup and Installation
### Clone the repository
```bash
!git clone https://github.com/ceyyg/FYP.git
%cd FYP
```

```bash
### Install the required dependencies
!pip install -r requirements.txt
```

### Dataset preparation
Due to the large size of the dataset containg thousands of images, the project used the FairFace dataset available as a zip file in Google Drive
```bash
from google.colab import drive
drive.mount("/content/drive")
# Unzip to local storage
!unzip -q "/content/drive/MyDrive/dataset.zip" -d "/content/"
```

## Project Structure
``` text
FYP/
├── src/
│   ├── main.py              # Main entry point (Coordinates experiments)
│   ├── train.py             # Training loop logic
│   ├── data.py              # Dataset & DataLoader initialization
│   ├── paths.py             # Global path configurations
│   ├── gridsearch.py        # Hyperparameter tuning logic
│   ├── resnet.py            # Model architecture (ResNet18)
│   ├── checkpoints.py       # State saving/loading logic
│   └── trial.py             # Initial data exploration & unzipping
├── results/                 # Local results cache
│   ├── matrices             # Raw confusion matrices
│   ├── raw_metrics          # Fairness metrics per run
│   ├── training_logs        # Loss/ Accuracy per epoch
│   ├── fairface_results.csv # Data Dictionary
├── tests/
│   ├── unittest.py          # Testing individual components
│   ├── integration.py       # Testing the comined components together
│   ├── system.py            # Testing the end to end pipeline
└── requirements.txt         # Python dependencies
```

## Running Experiment
The file runs all the related grid search, training and evaluation checkpointing every output to saved files in Drive.
```bash
!python src/main.py
```

## Evaluation Subgroup
The model evaluates fairness across 21 intersections for the gender prediction task, including:
Race: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino_Hispanic.
Age Brackets: Young, Middle and Young

Gender: Male, Female.
## Output files
The script generates several files in your SAVE_FOLDER (defined in src/paths.py):
```text
results_log.csv: Summary of all optimizer/seed combinations.
*_subgroups.csv: Detailed fairness metrics for all 21 subgroups.
*_history.csv: Epoch-by-epoch loss and accuracy curves.
*_swats_log.csv: (For SWATS only) Epoch of transition.
```

## Testing
To ensure mathematical integrity and reproducibility, this project includes a comprehensive testing suite divided into three tiers:

1. **Unit Testing:** Validates core logic including age-label preprocessing, subgroup stratification, and fairness metrics (ECE score and Accuracy Gap).
2. **Integration Testing:** Verifies the seamless interaction between the ResNet18 model, the data loaders, and the checkpoint system (ensuring 100% weight persistence).
3. **System Testing:** Confirms **Stochastic Parity**. By synchronizing random seeds (e.g., Seed 777), the system ensures that all experiments are 100% reproducible across different environments.


The project is for academic purpose as a part of the Final Year Project.
### Author: Celina Gurung