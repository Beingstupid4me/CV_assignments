import os
import torch
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from PIL import Image

# ✅ (1a) Initialize Weights & Biases (WandB)
wandb.init(project="russian-wildlife-cnn")

# ✅ (1a) Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ✅ (1a) Define data directory
data_dir = "CV//Cropped_final"

# ✅ (1a) Define class label mapping
class_mapping = {
    'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3,
    'brown_bear': 4, 'dog': 5, 'roe_deer': 6, 'sika_deer': 7,
    'wild_boar': 8, 'people': 9
}

# ✅ (1a) Load image paths and labels
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

# ✅ (1a) Stratified Split (80% Train, 20% Validation)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
train_idx, val_idx = next(splitter.split(image_paths, labels))

train_paths, train_labels = image_paths[train_idx], labels[train_idx]
val_paths, val_labels = image_paths[val_idx], labels[val_idx]

# ✅ (1a) Custom Dataset Class
class WildlifeDataset(Dataset):
    def _init_(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def _len_(self):
        return len(self.image_paths)
    
    def _getitem_(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # Open Image
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)  # ✅ Ensure labels are long

# ✅ (1b) Define Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ (1b) Create Dataset Instances
train_dataset = WildlifeDataset(train_paths, train_labels, transform=train_transform)
val_dataset = WildlifeDataset(val_paths, val_labels, transform=val_transform)

# ✅ (1b) Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ (2a) Define CNN Model with 3 Conv Layers
class CNN(nn.Module):
    def _init_(self):
        super(CNN, self)._init_()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 32 Filters
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # MaxPool (4x4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 64 Filters
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool (2x2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 128 Filters
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool (2x2, stride=2)
        self.fc = nn.Linear(128 * 14 * 14, 10)  # Fully Connected Layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ✅ (2b) Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ (2b) Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    wandb.log({"train_loss": train_loss, "train_acc": train_acc})

wandb.finish()

# ✅ (2d) Validation
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ (2d) Compute Metrics
val_acc = accuracy_score(all_labels, all_preds)
val_f1 = f1_score(all_labels, all_preds, average="weighted")
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")

# ✅ (2d) Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_mapping.keys(), yticklabels=class_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ✅ (2e) Log Confusion Matrix to WandB
wandb.init(project="russian-wildlife-cnn")
wandb.log({"confusion_matrix": wandb.Image(plt)})
wandb.finish()