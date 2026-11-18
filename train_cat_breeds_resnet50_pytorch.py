"""
Training ResNet50 untuk Klasifikasi 12 Ras Kucing (PyTorch Version)
Referensi: JIEET Volume 08 Nomor 01, 2024
Arsitektur: ResNet50 dengan Transfer Learning
Optimizer: SGD (Stochastic Gradient Descent)
Dataset: 12 Ras Kucing (Abyssinia, Bengal, Bombay, Birman, British Shorthair, Egyptian, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# Konfigurasi sesuai JIEET 2024
DATASET_DIR = "dataset_cat_breeds"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10  # Sesuai JIEET 2024
LEARNING_RATE = 0.01  # Learning rate untuk SGD
MOMENTUM = 0.9  # Momentum untuk SGD

# 12 Ras Kucing sesuai JIEET 2024
CAT_BREEDS = [
    "Abyssinia",
    "Bengal", 
    "Bombay",
    "Birman",
    "British Shorthair",
    "Egyptian",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx"
]

print("=" * 80)
print("TRAINING RESNET50 - KLASIFIKASI 12 RAS KUCING (PYTORCH)")
print("Referensi: JIEET Volume 08 Nomor 01, 2024")
print("=" * 80)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 1. Data Augmentation dan Loading
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("[INFO] Memuat dataset...")
full_dataset = ImageFolder(DATASET_DIR, transform=data_transforms['train'])

# Split 80% training, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Update transforms for validation set
val_dataset.dataset = ImageFolder(DATASET_DIR, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Training samples: {len(train_dataset)}")
print(f"[INFO] Validation samples: {len(val_dataset)}")
print(f"[INFO] Classes: {len(full_dataset.classes)}")
print(f"[INFO] Class names: {full_dataset.classes}")

# 2. Membangun Arsitektur ResNet50 dengan Transfer Learning
print("[INFO] Membangun Arsitektur ResNet50 Transfer Learning...")

# Load ResNet50 pre-trained on ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze backbone layers
for param in model.parameters():
    param.requires_grad = False

# Modify final layer untuk 12 kelas
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(CAT_BREEDS))
)

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# SGD optimizer (sesuai JIEET 2024)
optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

print(model)

# 3. Training Loop
train_losses = []
train_accs = []
val_losses = []
val_accs = []

print(f"[INFO] Memulai training model ({EPOCHS} epochs)...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# 4. Simpan Model
torch.save(model.state_dict(), "cat_breeds_resnet50_pytorch.pth")
print("[INFO] Model disimpan ke 'cat_breeds_resnet50_pytorch.pth'")

# 5. Visualisasi Training History
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_accs, label='Training Accuracy')
plt.plot(range(1, EPOCHS+1), val_accs, label='Validation Accuracy')
plt.title('Training and Validation Accuracy (ResNet50)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss (ResNet50)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('cat_breeds_resnet50_pytorch_training_curve.png', dpi=300, bbox_inches='tight')
print("[INFO] Grafik training disimpan ke 'cat_breeds_resnet50_pytorch_training_curve.png'")

# 6. Evaluasi Final
print("\n" + "=" * 80)
print("HASIL EVALUASI FINAL")
print("=" * 80)
print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")

print("\nTarget JIEET 2024:")
print(f"Training Accuracy: 99.17%")
print(f"Validation Accuracy: 88.13%")
print(f"Training Loss: 0.3760")
print(f"Validation Loss: 0.7857")

print("\n[INFO] Training selesai!")
