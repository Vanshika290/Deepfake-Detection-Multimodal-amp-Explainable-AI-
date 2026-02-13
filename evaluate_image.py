import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from train_image import build_model, IMAGE_SIZE, DEVICE

DATA_ROOT = os.path.join('dataset', 'images')

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def load_loader(batch_size=32):
    ds = ImageFolder(DATA_ROOT, transform=get_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

def evaluate(model_path='deepfake_image_best.pth'):
    print(f'Loading image model from {model_path} on {DEVICE}')
    model = build_model()
    if not os.path.exists(model_path):
        print('Model file not found:', model_path)
        return
    ck = torch.load(model_path, map_location=DEVICE)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'])
    else:
        model.load_state_dict(ck)

    model.eval()
    loader = load_loader()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            scores = probs[:,1].cpu().numpy().tolist()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds)
            y_score.extend(scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')

    print('Image Evaluation results:')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1-score:  {f1:.4f}')
    print(f'  ROC AUC:   {auc:.4f}')

if __name__ == "__main__":
    evaluate()
