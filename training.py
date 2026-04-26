"""
Trains Resnet-18 on CIFAR-10

Outputs:
    - resnet18_cifar10_baseline.pth (full model weights, FP32)
    - training_log.csv (per epoch train loss, test loss, test accuracy)
"""

import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 2
SAVE_PATH = "resnet18_cifar10_baseline.pth"
LOG_PATH = "training_log.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed = 42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

set_seed(42)

# CIFAR-10 images are 32 by 32. augmentation techniques applied for this dataset:
#   random crop with padding 
#   random horizontal flip
#   normalize to per channel mean/std computed over training set

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f"Training samples: {len(train_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")
print(f"Batches/epoch: {len(train_loader)}")

# torchvision's ResNet-18 expects 224 by 224 ImageNet inputs by default
# For CIFAR-10 (32x32), swap the aggressive 7x7 conv + maxpool stem with a 3x3 conv stem

model = torchvision.models.resnet18(weights=None, num_classes=10)

# Replace the ImageNet stem with a CIFAR-friendly stem
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.bn1 = nn.BatchNorm2d(64)
model.maxpool = nn.Identity()  # remove the 3×3 maxpool

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=LEARNING_RATE,
    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)

# Cosine annealing decays LR smoothly from 0.1 to ~0, standard practice for CIFAR-10 to hit 93%
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

print(f"Starting training: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
print(f"{'='*60}\n")

best_acc = 0.0
log_rows = []
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]["lr"]

    # log
    log_rows.append({
        "epoch": epoch,
        "train_loss": f"{train_loss:.4f}",
        "train_acc": f"{train_acc:.2f}",
        "test_loss": f"{test_loss:.4f}",
        "test_acc": f"{test_acc:.2f}",
        "lr": f"{current_lr:.6f}",
        "epoch_sec": f"{epoch_time:.1f}",
    })

    print(
        f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.2f}% | "
        f"LR: {current_lr:.5f} | "
        f"{epoch_time:.1f}s"
    )

    # save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc,
            "test_loss": test_loss,
        }, SAVE_PATH)
        print(f"  ↑ New best model saved ({test_acc:.2f}%)")

total_time = time.time() - start_time

with open(LOG_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    writer.writeheader()
    writer.writerows(log_rows)

model_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)

print(f"Training complete")
print(f" Total time: {total_time/60:.1f} minutes")
print(f" Best test acc: {best_acc:.2f}%")
print(f" Model size (FP32): {model_size_mb:.2f} MB")
print(f" Weights: {SAVE_PATH}")
print(f" Training log:{LOG_PATH}")
