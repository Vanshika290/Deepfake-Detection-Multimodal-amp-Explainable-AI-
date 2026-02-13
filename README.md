# Deepfake Detection — local training & API

This repo includes scripts for training and evaluating image and video deepfake detectors, plus a simple FastAPI server and frontend.

Quick start

1. Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt
```

2. Train video model (uses `dataset/SDFVD/SDFVD/videos_real` and `videos_fake`):

```bash
python train_video.py
```

3. Train image model (streaming example is in `train.py`) or adapt to local dataset.

4. Run API server:

```bash
python app.py
# then open the frontend at frontend/index.html (or run `npm install && npm run dev` in the frontend folder)
```

5. Evaluate the saved video model:

```bash
python evaluate.py
```

Notes

- `app.py` exposes `/predict` for videos and `/predict_image` for images.
- Frontend `frontend/src/App.jsx` posts to these endpoints and shows JSON results.

- For best accuracy, train the video model with `train_video.py` for many epochs on a GPU.

## GitHub Setup

This repository is configured to be lightweight for GitHub.
1. The `dataset/` folder is ignored (add your own data if needed).
2. Large model checkpoints are ignored, except `video_model_best.pth`.
3. Virtual environment (`venv/`) is ignored.

To run this project after cloning:
1. Install requirements: `pip install -r requirements.txt`
2. Download or place `video_model_best.pth` in the root (if provided separately).
3. Run `python app.py`.
