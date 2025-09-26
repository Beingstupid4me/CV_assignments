import os
import torch
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

# Initialize Weights & Biases
wandb.init(project="russian-wildlife")

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define data directory
data_dir = "CV//Cropped_final"

# Define class label mapping
class_mapping = {
    'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3,
    'brown_bear': 4, 'dog': 5, 'roe_deer': 6, 'sika_deer': 7,
    'wild_boar': 8, 'people': 9
}

# Create list of image paths and labels
image_paths = []
labels = []
print("Loading Data...")
for class_name, class_label in class_mapping.items():
    class_folder = os.path.join(data_dir, class_name)
    if os.path.isdir(class_folder):
        for img_file in os.listdir(class_folder):
            if img_file.endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(class_folder, img_file))
                labels.append(class_label)

# Convert to numpy arrays for stratified split
image_paths = np.array(image_paths)
labels = np.array(labels)
print(f"Total Images: {len(image_paths)}")
# Perform Stratified Split (80% Train, 20% Validation)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
train_idx, val_idx = next(splitter.split(image_paths, labels))

train_paths, train_labels = image_paths[train_idx], labels[train_idx]
val_paths, val_labels = image_paths[val_idx], labels[val_idx]


class WildlifeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  
        
        if self.transform:
            image = self.transform(image)

        return image, label


# Define Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create Dataset Instances
train_dataset = WildlifeDataset(train_paths, train_labels, transform=train_transform)
val_dataset = WildlifeDataset(val_paths, val_labels, transform=val_transform)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Function to Plot Class Distribution
def plot_class_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8, 5))
    plt.bar([list(class_mapping.keys())[i] for i in unique], counts, color='c')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()


# Visualize Class Distribution
plot_class_distribution(train_labels, "Training Set Distribution")
plot_class_distribution(val_labels, "Validation Set Distribution")

wandb.finish()
