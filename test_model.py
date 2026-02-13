
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

# Configuration
MODEL_PATH = "deepfake_model_epoch_1.pth" # Start testing with the first checkpoint
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Fake', 'Real'] # 0: Fake, 1: Real (Standard labeled datasets usually follow this, but need verification)

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except ImportError:
        model = models.resnet18(pretrained=True)
        
    # Modify final layer to match training
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model file '{model_path}' not found.")
        return None
        
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_path):
    import io
    import traceback
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        # Try to read file as bytes first to handle some filesystem/OneDrive quirks
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        label = CLASSES[predicted_class.item()]
        conf_score = confidence.item() * 100
        
        # Determine color for terminal output? (Simple text for now)
        print(f"\nResult for: {os.path.basename(image_path)}")
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {conf_score:.2f}%")
        print(f"Probabilities: Fake: {probabilities[0][0]*100:.2f}%, Real: {probabilities[0][1]*100:.2f}%")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # traceback.print_exc() # Uncomment if you want full stack trace




if __name__ == "__main__":
    from datasets import load_dataset
    import random
    import glob
    
    # Auto-detect latest model
    model_files = glob.glob("deepfake_model_*.pth")
    if not model_files:
        print("No model files found!")
        sys.exit(1)
        
    # Sort to find the best one: Prefer 'final', then highest epoch
    # Heuristic: 'final' > 'epoch_5' > 'epoch_1' ...
    def sort_key(f):
        if 'final' in f:
            return 9999
        # Extract epoch number
        parts = f.replace('.pth', '').split('_')
        if len(parts) > 0 and parts[-1].isdigit():
            return int(parts[-1])
        return 0
        
    latest_model_path = sorted(model_files, key=sort_key, reverse=True)[0]
    print(f"Using model: {latest_model_path}")
    
    model = load_model(latest_model_path)
    if not model:
        sys.exit(1)


    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
        
        # Check if it's a directory or file
        if os.path.isdir(path_arg):
            print(f"\nScanning directory: {path_arg}")
            # simple glob for common image formats
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(path_arg, ext)))
                # Try uppercase too just in case (Windows is case-insensitive usually, but good practice)
                image_files.extend(glob.glob(os.path.join(path_arg, ext.upper())))
            
            # Remove duplicates
            image_files = sorted(list(set(image_files)))
            
            if not image_files:
                print("No image files (jpg, jpeg, png, webp) found in this directory.")
            else:
                print(f"Found {len(image_files)} images. Processing first 10...")
                for img_f in image_files[:10]:
                    print("-" * 30)
                    predict_image(model, img_f)
                if len(image_files) > 10:
                    print(f"...and {len(image_files)-10} more.")
                    
        elif os.path.exists(path_arg):
            predict_image(model, path_arg)
        else:
            print(f"Error: File or directory not found: {path_arg}")
            print("Note: If you are trying to access a connected phone/device, please copy the image to your Desktop first.")
            
    else:
        print("\nNo image path provided. Fetching a random sample from dataset...")
        try:
            # Re-used code from previous step - simplified here to reduce diff size in thought
            dataset = load_dataset("saakshigupta/deepfake-detection-dataset-v3", streaming=True)
            ds = dataset['test'] if 'test' in dataset else dataset['train']
            
            iterator = iter(ds)
            skip = random.randint(0, 10)
            for _ in range(skip):
                next(iterator)
            sample = next(iterator)
            
            img = sample['image']
            true_label = CLASSES[sample['label']]
            
            print(f"Sample fetched. True Label: {true_label}")
            
            transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
            pred_label = CLASSES[pred_idx.item()]
            print(f"Prediction: {pred_label} ({conf.item()*100:.2f}%)")
            
            if pred_label == true_label:
                print("✅ Correct!")
            else:
                print("❌ Incorrect.")
                
        except Exception as e:
            print(f"Error fetching sample: {e}")



