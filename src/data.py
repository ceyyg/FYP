import torch, os, random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from src.trial import dataset_dir, train, test
from src.trial import collapse_age

for df in (train, test):
  df['age_group'] = df["age"].apply(collapse_age)

# Split the training into 80/20 for training and validation set and stratifying by gender
train, val = train_test_split(
    train,
    test_size=0.2,
    random_state=99,
    stratify=train["gender"]
)

# Reset indices to ensure clean iterations
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

# Standarizing to match ImageNet 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training Transformations
train_trs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Validation Transformations
val_trs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Test Transformations
test_trs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class FairFace(Dataset):
  """
  Custom Dataset class for retrieval of Gender, Race, Age.
  Tested in Unit Testing to output consistent tensor shapes.
  """
  def __init__(self, df, root_dir, transform=None):
    self.df = df.reset_index(drop=True)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    # Load and convert the images to RGB to ensure chanel consistency
    row = self.df.iloc[idx]
    image_path = os.path.join(self.root_dir, row["file"])
    image = Image.open(image_path).convert("RGB")
    label = 1 if row["gender"] == "Male" else 0 # Gender label: 0 for Female, 1 for Male

    if self.transform:
      image = self.transform(image)

    race = row["race"]
    age_group = collapse_age(row["age"])
    return image, label, race, age_group
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

pin_memory = device.type == "cuda"

train_dataset = FairFace(train, dataset_dir, train_trs)
val_dataset   = FairFace(val, dataset_dir, val_trs)
test_dataset  = FairFace(test, dataset_dir, test_trs)

print("Train dataset:", len(train_dataset))
print("Val dataset:", len(val_dataset))
print("Test dataset:", len(test_dataset))

batch_size = 32
def data_loaders(seed):
  """"
  System Testing: Handles data shuffling
  Seed synchronisation ensures reproducible results across all runs.
  """
  generator = torch.Generator().manual_seed(seed)

  train_loader = DataLoader(
      train_dataset,
      batch_size = batch_size,
      shuffle=True,
      pin_memory = pin_memory,
      generator = generator)

  val_loader = DataLoader(
      val_dataset,
      batch_size = batch_size,
      shuffle = False,
      pin_memory = pin_memory)

  test_loader = DataLoader(
      test_dataset,
      batch_size = batch_size,
      shuffle = False,
      pin_memory = pin_memory)

  return train_loader, val_loader, test_loader


def audit_dataset(dataset, num_samples=5):
    """
    Verifies that the Index, Filename, and the full Intersectional 
    labels (Race, Gender, Age) are perfectly synchronized.
    """
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
   audit_dataset(train_dataset, num_samples = 5)




