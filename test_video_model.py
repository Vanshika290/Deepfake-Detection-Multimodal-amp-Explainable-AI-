import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os
import numpy as np
from pathlib import Path

# Configuration
IMAGE_SIZE = (224, 224)
FRAMES_PER_VIDEO = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_ROOT = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD"

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

def get_transforms():
    """Transforms for test (no augmentation)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_frames(video_path, frames_per_video=10):
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
    frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    
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
    while len(frames) < frames_per_video:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    return frames[:frames_per_video]

def predict_video(model, video_path, transform):
    """Predict if a video is fake or real"""
    # Extract frames
    frames = extract_frames(video_path, FRAMES_PER_VIDEO)
    
    if frames is None:
        return None, None
    
    # Transform frames
    frames_tensor = [transform(frame) for frame in frames]
    frames_tensor = torch.stack(frames_tensor).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    frames_tensor = frames_tensor.to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(frames_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def test_model(model_path="video_model_best.pth"):
    """Test the trained model on the validation set"""
    print(f"Using device: {DEVICE}")
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model file {model_path} not found!")
        return
    
    model.eval()
    transform = get_transforms()
    
    # Test on some videos
    fake_dir = os.path.join(VIDEO_ROOT, 'videos_fake')
    real_dir = os.path.join(VIDEO_ROOT, 'videos_real')
    
    print("\n" + "=" * 70)
    print("Testing on sample videos...")
    print("=" * 70)
    
    # Test fake videos
    if os.path.exists(fake_dir):
        fake_videos = [f for f in os.listdir(fake_dir) if f.endswith('.mp4')][:5]
        print("\n[VIDEO] Testing FAKE videos:")
        print("-" * 70)
        
        fake_correct = 0
        for video_file in fake_videos:
            video_path = os.path.join(fake_dir, video_file)
            predicted_class, confidence = predict_video(model, video_path, transform)
            
            if predicted_class is not None:
                label = "FAKE" if predicted_class == 1 else "REAL"
                is_correct = predicted_class == 1
                fake_correct += is_correct
                
                status = "[OK]" if is_correct else "[FAIL]"
                print(f"{status} {video_file:20s} | Predicted: {label:4s} | Confidence: {confidence:.2%}")
        
        print(f"\nFake videos accuracy: {fake_correct}/{len(fake_videos)} ({100*fake_correct/len(fake_videos):.1f}%)")
    
    # Test real videos
    if os.path.exists(real_dir):
        real_videos = [f for f in os.listdir(real_dir) if f.endswith('.mp4')][:5]
        print("\n[VIDEO] Testing REAL videos:")
        print("-" * 70)
        
        real_correct = 0
        for video_file in real_videos:
            video_path = os.path.join(real_dir, video_file)
            predicted_class, confidence = predict_video(model, video_path, transform)
            
            if predicted_class is not None:
                label = "FAKE" if predicted_class == 1 else "REAL"
                is_correct = predicted_class == 0
                real_correct += is_correct
                
                status = "[OK]" if is_correct else "[FAIL]"
                print(f"{status} {video_file:20s} | Predicted: {label:4s} | Confidence: {confidence:.2%}")
        
        print(f"\nReal videos accuracy: {real_correct}/{len(real_videos)} ({100*real_correct/len(real_videos):.1f}%)")
    
    print("\n" + "=" * 70)

def test_single_video(video_path, model_path="video_model_best.pth"):
    """Test a single video file"""
    print(f"Using device: {DEVICE}")
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model file {model_path} not found!")
        return
    
    model.eval()
    transform = get_transforms()
    
    print(f"\nTesting video: {video_path}")
    predicted_class, confidence = predict_video(model, video_path, transform)
    
    if predicted_class is not None:
        label = "FAKE" if predicted_class == 1 else "REAL"
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        
        if predicted_class == 1:
            print("[WARN]  This video appears to be a DEEPFAKE")
        else:
            print("[OK] This video appears to be REAL")
    else:
        print("Error: Could not process video")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test a single video
        video_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "video_model_best.pth"
        test_single_video(video_path, model_path)
    else:
        # Test on validation set
        test_model()
