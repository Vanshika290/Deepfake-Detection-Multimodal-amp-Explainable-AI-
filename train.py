
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from datasets import load_dataset
import time
import os


# Configuration for Full Training
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def train_full():
    print("Loading dataset (streaming)...", flush=True)
    dataset = load_dataset("saakshigupta/deepfake-detection-dataset-v3", streaming=True)
    train_ds = dataset['train']
    
    img_transforms = get_transforms()
    
    print("Building model...", flush=True)
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except ImportError:
        model = models.resnet18(pretrained=True)
        
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Setup mixed precision only for CUDA
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    
    print(f"Starting full training on {DEVICE} for {NUM_EPOCHS} epochs...", flush=True)
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---", flush=True)
        model.train()
        
        # Re-initialize iterator for each epoch
        iterator = iter(train_ds)
        
        step = 0
        running_loss = 0.0
        correct = 0
        total = 0
        
        while True:
            batch_images = []
            batch_labels = []
            
            try:
                # Manually fetch BATCH_SIZE items
                for _ in range(BATCH_SIZE):
                    item = next(iterator)
                    img = item['image']
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    tensor = img_transforms(img)
                    batch_images.append(tensor)
                    batch_labels.append(item['label'])
            except StopIteration:
                # If we got some items before stopping, process them. Otherwise break.
                if not batch_images:
                    break
            except Exception as e:
                print(f"Error loading batch: {e}")
                break
                
            if not batch_images:
                break
                
            # Stack
            inputs = torch.stack(batch_images).to(DEVICE)
            labels = torch.tensor(batch_labels).to(DEVICE)
            
            optimizer.zero_grad()
            
            if use_cuda:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Stats
            step += 1
            loss_val = loss.item()
            running_loss += loss_val
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if step % 10 == 0:
                avg_loss = running_loss / 10
                acc = 100. * correct / total
                print(f"Epoch {epoch+1} | Step {step} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%", flush=True)
                running_loss = 0.0
                correct = 0
                total = 0

            # Stop manually if dataset is exhausted (StopIteration is caught above)
            # The inner loop logic handles the "last partial batch" case if StopIteration raised inside loop
            # But the try/except block above might break early. 
            # If len(batch_images) < BATCH_SIZE, it means we hit end of dataset.
            if len(batch_images) < BATCH_SIZE:
                 break
        
        # Save checkpoint
        checkpoint_name = f"deepfake_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_name)
        print(f"Saved checkpoint: {checkpoint_name}", flush=True)
        
    total_duration = time.time() - start_time
    print(f"Full training finished in {total_duration:.2f}s", flush=True)
    
    # Save final model
    torch.save(model.state_dict(), "deepfake_model_final.pth")
    print("Final model saved to deepfake_model_final.pth")

if __name__ == "__main__":
    train_full()

