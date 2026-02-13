import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

# Config
BATCH_SIZE = 32
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join("dataset", "images")  # expects images/real and images/fake

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

def build_model(num_classes=2, freeze_backbone=True):
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def load_datasets(root=DATA_ROOT, val_fraction=0.2):
    if not os.path.exists(root):
        print(f"Image dataset folder not found at {root}. Expected structure: dataset/images/real and dataset/images/fake")
        return None, None

    full = datasets.ImageFolder(root, transform=get_transforms(train=True))
    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val])

    # Replace transforms on validation subset
    val_ds.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader

def train():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = load_datasets()
    if train_loader is None:
        return

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    scaler = torch.amp.GradScaler() if DEVICE.type == 'cuda' else None

    best_acc = 0.0
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        preds = []
        trues = []

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

        train_acc = accuracy_score(trues, preds) if trues else 0.0

        # Validation
        model.eval()
        v_preds = []
        v_trues = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                v_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
                v_trues.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(v_trues, v_preds) if v_trues else 0.0
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {running_loss:.4f}")

        ck = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }
        torch.save(ck, f"deepfake_image_epoch_{epoch+1}.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ck, "deepfake_image_best.pth")

    total = time.time() - start
    print(f"Image training completed in {total:.2f}s. Best val acc: {best_acc:.4f}")

if __name__ == '__main__':
    train()
