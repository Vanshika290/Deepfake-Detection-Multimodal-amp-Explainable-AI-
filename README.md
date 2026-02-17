

https://github.com/user-attachments/assets/858fb275-cdc0-4e8c-9907-a1f7ae9a9604


<h1 align="center" style="font-size: 3em;">🔍 Multimodal Deepfake Detection using Explainable AI</h1>

<p align="center">
An AI-powered system to detect manipulated media using audio-visual fusion and transparent explainability
</p>

---

## 📌 Overview

Deepfake technology has made it increasingly difficult to trust digital media. This project introduces a **multimodal deepfake detection system** that analyzes both **video and audio inputs** to classify media as real or fake. To ensure transparency, **Explainable AI (XAI)** techniques are used to highlight manipulated regions and features.

---

## 🎯 Problem Statement

The rise of AI-generated fake videos and voices has created serious threats in misinformation, fraud, and digital forensics. Most existing solutions are either single-modal or black-box models. This project addresses these challenges by combining **audio-visual analysis** with **interpretable deep learning**.

---

## 🚀 Key Features

* 🎥 Video-based deepfake detection using CNNs
* 🔊 Audio deepfake detection via spectrogram analysis
* 🔗 Multimodal fusion for higher accuracy
* 🧠 Explainable AI using Grad-CAM and saliency maps
* 📊 Confidence score with visual explanations
* 🌐 Scalable backend for real-world deployment

---

## 🧠 System Architecture

```
Video Input
   ↓
Frame & Audio Extraction
   ↓
Video CNN (Face Analysis)      Audio CNN (Spectrogram)
           ↓                    ↓
         Feature Fusion (Multimodal)
                    ↓
              Classification
                    ↓
           Explainable AI (XAI)
                    ↓
              Final Prediction
```

---

## ⚙️ Tech Stack

* **Language:** Python
* **Deep Learning:** PyTorch
* **Computer Vision:** OpenCV
* **Audio Processing:** Librosa
* **Explainability:** Grad-CAM, Saliency Maps
* **Backend:** Flask / FastAPI
* **Frontend:** HTML, CSS, JavaScript / React
* **Deployment:** AWS / Local

---

## 📂 Project Structure

```
deepfake-detection/
│
├── data/
│   ├── video/
│   └── audio/
│
├── models/
│   ├── video_model.py
│   ├── audio_model.py
│   └── fusion_model.py
│
├── explainability/
│   ├── grad_cam.py
│   └── audio_saliency.py
│
├── utils/
│   ├── frame_extraction.py
│   └── audio_extraction.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Datasets Used

* **FaceForensics++** – Video deepfake dataset
* **DFDC (Facebook Deepfake Detection Challenge)**
* **ASVspoof 2019** – Audio deepfake dataset

> Due to large dataset sizes, only selected subsets are used.

---

## 🧪 How It Works

1. User uploads a video file
2. Frames and audio are extracted
3. Video and audio models analyze inputs
4. Multimodal fusion combines predictions
5. Explainable AI highlights suspicious regions
6. Final result with confidence score is displayed

---

## 🔐 Use Cases

* 📰 Media and news verification
* ⚖️ Digital forensics and law enforcement
* 🛡️ Fraud and identity protection
* 📱 Social media content moderation
* 🗳️ Election and misinformation security

---

## 🔮 Future Enhancements

* Real-time deepfake detection
* Transformer-based video models
* Attention-based fusion techniques
* Browser and video-call integrations
* Cloud-based API service

---

## 👩‍💻 Author

**Vanshika Saxena**
B.Tech Computer Science
AI & Machine Learning Enthusiast

---

## ⭐ Acknowledgements

* Open-source AI and ML community
* Research papers on deepfake detection
* Publicly available datasets

---

<p align="center">⭐ If you find this project useful, consider giving it a star!</p>
