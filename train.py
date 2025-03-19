import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define transformations with data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformations for validation and test
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset, analyse and apply stratified split
def load_dataset():
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    targets = np.array(dataset.targets)
    
    # Exploratory Data Analysis (EDA)
    class_counts = np.bincount(targets)
    class_labels = dataset.classes
    # plt.figure(figsize=(10, 5))
    # sns.barplot(x=class_labels, y=class_counts)
    # plt.xlabel("Class Labels")
    # plt.ylabel("Number of Samples")
    # plt.title("Class Distribution in CIFAR-10 Dataset")
    # plt.xticks(rotation=45)
    # plt.show()
    print("Dataset EDA:")
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution: {dict(zip(class_labels, class_counts))}")

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(stratified_split.split(np.zeros(len(targets)), targets))
    train_targets = targets[train_idx]
    temp_targets = targets[temp_idx]
    
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(stratified_split.split(np.zeros(len(temp_targets)), temp_targets))
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, temp_idx[val_idx])
    test_dataset = torch.utils.data.Subset(dataset, temp_idx[test_idx])
    
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)
# Freeze all layers for transfer learning.
for param in model.parameters():
    param.requires_grad = False 

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define loss, optimizer, scaler for mixed precision, and early stopping
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()
patience, best_loss, counter = 2, np.inf, 0  # Early stopping parameters

# Train model
def train_model(model, train_loader, val_loader, epochs=5):
    global best_loss, counter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        # Validation step
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, average='macro')
        rec = recall_score(val_labels, val_preds, average='macro')
        print(f"Validation Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Val Loss: {val_loss/len(val_loader)}")
        model.train()
        
        # Ensure the 'models' directory exists
        os.makedirs("models", exist_ok=True)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "models/resnet50_cifar10.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

train_loader, val_loader, test_loader = load_dataset()
train_model(model, train_loader, val_loader)
print("Model training complete and saved to models/resnet50_cifar10.pth")
