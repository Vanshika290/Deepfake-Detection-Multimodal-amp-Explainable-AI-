
import zipfile
import os
import shutil

zip_path = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\SDFVD Small-scale Deepfake Forgery Video Dataset\SDFVD Small-scale Deepfake Forgery Video Dataset\SDFVD.zip"
extract_path = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)
    print(f"Created {extract_path}")

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete.")

# Check extracted structure
video_root = os.path.join(extract_path, "SDFVD")
if os.path.exists(video_root):
    print(f"Contents of {video_root}: {os.listdir(video_root)}")
else:
    print(f"Could not find extracted root dir at {video_root}")
