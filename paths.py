import os

# Global Path configurations
dataset_dir = "/content/dataset"
checkpts = "/content/drive/MyDrive/fairface_ckpts"
result_path = "/content/drive/MyDrive/fairface_results.csv"
gs_path = "/content/drive/MyDrive/fairface_gs.json"
save_folder = "/content/drive/MyDrive/fairface_project/results"

# Ensure directories exist for checkpointing and results
os.makedirs(save_folder, exist_ok=True)
os.makedirs(checkpts, exist_ok=True)