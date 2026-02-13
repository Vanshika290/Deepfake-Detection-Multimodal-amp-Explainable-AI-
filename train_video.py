import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from pathlib import Path
import time
from sklearn.model_selection import train_test_split

# Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
IMAGE_SIZE = (224, 224)
FRAMES_PER_VIDEO = 10  # Extract 10 frames from each video
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_ROOT = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD"

class VideoFrameDataset(Dataset):
    """Dataset that extracts frames from videos"""
    
    def __init__(self, video_paths, labels, transform=None, frames_per_video=10):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        """Extract evenly spaced frames from a video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Calculate frame indices to extract (evenly spaced)
        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame extraction fails, use a black frame
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        
        # If we didn't get enough frames, pad with the last frame
        while len(frames) < self.frames_per_video:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.frames_per_video]
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if frames is None:
                # Return a dummy tensor if video couldn't be read
                frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.frames_per_video)]
            
            # Transform frames
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            
            # Stack frames along a new dimension: (frames, channels, height, width)
            frames_tensor = torch.stack(frames)
            
        except Exception as e:
            print(f"Warning: Error processing video {video_path}: {e}")
            # Return dummy data on error
            dummy_frame = torch.zeros((3, 224, 224))
            frames_tensor = torch.stack([dummy_frame for _ in range(self.frames_per_video)])
        
        return frames_tensor, label

def get_transforms():
    """Data augmentation and normalization"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_test_transforms():
    """Transforms for validation/test (no augmentation)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class DeepfakeDetector(nn.Module):
    """Model that processes multiple frames from a video"""
    
    def __init__(self, num_frames=10):
        super(DeepfakeDetector, self).__init__()
        
        # Load pretrained ResNet18
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            self.feature_extractor = models.resnet18(weights=weights)
        except ImportError:
            self.feature_extractor = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Freeze early layers for faster training
        for param in list(self.feature_extractor.parameters())[:-20]:
            param.requires_grad = False
        
        # LSTM to process temporal information
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, 
                           batch_first=True, dropout=0.3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.size()
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features from each frame
        features = self.feature_extractor(x)  # (batch*frames, 512, 1, 1)
        features = features.view(batch_size, num_frames, -1)  # (batch, frames, 512)
        
        # Process temporal sequence with LSTM
        lstm_out, _ = self.lstm(features)  # (batch, frames, 256)
        
        # Use the last LSTM output
        last_output = lstm_out[:, -1, :]  # (batch, 256)
        
        # Classify
        output = self.classifier(last_output)  # (batch, 2)
        
        return output

def load_video_dataset():
    """Load video paths and labels"""
    fake_dir = os.path.join(VIDEO_ROOT, 'videos_fake')
    real_dir = os.path.join(VIDEO_ROOT, 'videos_real')
    
    video_paths = []
    labels = []
    
    # Load fake videos (label = 1)
    if os.path.exists(fake_dir):
        for video_file in os.listdir(fake_dir):
            if video_file.endswith('.mp4'):
                video_paths.append(os.path.join(fake_dir, video_file))
                labels.append(1)  # Fake
    
    # Load real videos (label = 0)
    if os.path.exists(real_dir):
        for video_file in os.listdir(real_dir):
            if video_file.endswith('.mp4'):
                video_paths.append(os.path.join(real_dir, video_file))
                labels.append(0)  # Real
    
    print(f"Total videos loaded: {len(video_paths)}")
    print(f"Fake videos: {sum(labels)}, Real videos: {len(labels) - sum(labels)}")
    
    return video_paths, labels

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (frames, labels) in enumerate(dataloader):
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        use_cuda = device.type == 'cuda'
        
        if use_cuda and scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(frames)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 5 == 0:
            avg_loss = running_loss / 5
            acc = 100. * correct / total
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%", flush=True)
            running_loss = 0.0
    
    epoch_acc = 100. * correct / total
    return epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    print(f"Using device: {DEVICE}")
    print(f"Training configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Frames per video: {FRAMES_PER_VIDEO}")
    print()
    
    # Load dataset
    print("Loading video dataset...")
    video_paths, labels = load_video_dataset()
    
    if len(video_paths) == 0:
        print("Error: No videos found!")
        return
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train videos: {len(train_paths)}, Validation videos: {len(val_paths)}")
    print()
    
    # Create datasets
    train_dataset = VideoFrameDataset(train_paths, train_labels, 
                                     transform=get_transforms(), 
                                     frames_per_video=FRAMES_PER_VIDEO)
    val_dataset = VideoFrameDataset(val_paths, val_labels, 
                                   transform=get_test_transforms(), 
                                   frames_per_video=FRAMES_PER_VIDEO)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    
    # Create model
    print("Building model...")
    model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=2)
    
    # Mixed precision training
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    
    print("Starting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save checkpoint
        checkpoint_path = f"video_model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, "video_model_best.pth")
            print(f"  ⭐ New best model! Val Accuracy: {val_acc:.2f}%")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), "video_model_final.pth")
    print("Final model saved to video_model_final.pth")

if __name__ == "__main__":
    main()
