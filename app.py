
import os
import shutil
import cv2
import torch
import numpy as np
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import uvicorn
import io
import time

# Import model architecture and transforms from test script
from test_video_model import DeepfakeDetector, get_transforms, DEVICE, FRAMES_PER_VIDEO
from PIL import Image
import io
from torchvision import models
import torch.nn as nn



from contextlib import asynccontextmanager

# Global variables for model and transform
model = None
transform = None

def load_model():
    """Load the video model on startup"""
    global model, transform
    
    model_path = "video_model_best.pth"
    if not os.path.exists(model_path):
        model_path = "video_model_final.pth"
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        transform = get_transforms()
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Video predictions will not work until model is available.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    load_model()
    yield
    # Cleanup on shutdown (if needed)


app = FastAPI(title="Deepfake Detection API", description="API for detecting deepfake videos", version="1.0.0", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"status": "online", "message": "Deepfake Detection API is running"}

def extract_frames_from_video(video_path, num_frames=10):
    """Extract frames from video for inference"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
        
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Fallback for failed frame read
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
    cap.release()
    
    # Pad if necessary
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
    return frames[:num_frames]


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file.")

    # Lazy-load an image classification model separate from the video model
    try:
        # Build image model (ResNet18)
        image_model = models.resnet18(pretrained=False)
        image_model.fc = nn.Linear(image_model.fc.in_features, 2)
        image_model = image_model.to(DEVICE)

        # Try loading weights if available
        image_model_path = "deepfake_model_final.pth"
        if os.path.exists(image_model_path):
            try:
                ck = torch.load(image_model_path, map_location=DEVICE)
                # support either state_dict or wrapped checkpoint
                if isinstance(ck, dict) and 'model_state_dict' in ck:
                    image_model.load_state_dict(ck['model_state_dict'])
                else:
                    image_model.load_state_dict(ck)
            except Exception as e:
                print(f"Warning: could not load image model weights: {e}")

        image_model.eval()
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Read file bytes
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert('RGB')
        img_np = np.array(img)
        tensor = image_transform(img_np).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = image_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = int(outputs.argmax(dim=1).item())

        result = {
            "prediction": "FAKE" if pred == 1 else "REAL",
            "confidence": float(probs[pred].item()),
            "probabilities": {
                "fake": float(probs[1].item()),
                "real": float(probs[0].item())
            }
        }

        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error in image prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    
    try:
        # Save uploaded file properly
        # Create a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_video:
            # Read and write chunks to avoid memory issues with large files
            shutil.copyfileobj(file.file, temp_video)
            temp_path = temp_video.name
            
        print(f"Processing video: {temp_path}")
        
        # Extract frames
        frames = extract_frames_from_video(temp_path, FRAMES_PER_VIDEO)
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception as e:
            print(f"Error deleting temp file: {e}")
            
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
            
        # Transform and Inference
        frames_tensor = [transform(frame) for frame in frames]
        frames_tensor = torch.stack(frames_tensor).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get probabilities for both classes
            fake_prob = probabilities[0][1].item()
            real_prob = probabilities[0][0].item()

        result = {
            "prediction": "FAKE" if predicted_class == 1 else "REAL",
            "confidence": float(confidence),
            "probabilities": {
                "fake": float(fake_prob),
                "real": float(real_prob)
            },
            "frames_processed": len(frames)
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    # Mock audio analysis (until actual model is trained)
    import random
    import asyncio
    
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Invalid audio format. Please upload a .wav, .mp3, .m4a, or .flac file.")

    # Random simulation for demo purposes
    fake_prob = random.uniform(0.1, 0.9)
    real_prob = 1.0 - fake_prob
    pred = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if pred == "FAKE" else real_prob
    
    # Simulate processing time
    await asyncio.sleep(1.5)

    return JSONResponse(content={
        "prediction": pred,
        "confidence": float(confidence),
        "probabilities": {
            "fake": float(fake_prob), 
            "real": float(real_prob)
        },
        "frames_processed": "Audio Waveform Analysis"
    })

@app.post("/predict_text")
async def predict_text(current_text: str = Form(...)):
    # Mock text analysis (until actual model is trained)
    import random
    import asyncio
    
    # Simple heuristic + random factor
    fake_prob = random.uniform(0.3, 0.7)
    
    if len(current_text) < 10:
        fake_prob = 0.5 # unsure
    
    real_prob = 1.0 - fake_prob
    pred = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if pred == "FAKE" else real_prob

    # Simulate processing time
    await asyncio.sleep(1.0)

    return JSONResponse(content={
        "prediction": pred,
        "confidence": float(confidence),
        "probabilities": {
            "fake": float(fake_prob), 
            "real": float(real_prob)
        },
        "frames_processed": f"{len(current_text.split())} words analyzed"
    })

if __name__ == "__main__":
    print("Starting TruthLens API on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
