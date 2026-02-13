import os
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
VIDEO_ROOT = os.path.join(ROOT, 'dataset', 'SDFVD', 'SDFVD')
OUT_ROOT = os.path.join(ROOT, 'dataset', 'images')
FRAMES_PER_VIDEO = 5

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def extract_from_dir(src_dir, label):
    out_dir = os.path.join(OUT_ROOT, label)
    ensure_dir(out_dir)

    if not os.path.exists(src_dir):
        print(f'Source directory not found: {src_dir}')
        return 0

    videos = [f for f in os.listdir(src_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    count = 0
    for v in videos:
        vpath = os.path.join(src_dir, v)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print('Could not open', vpath)
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        indices = np.linspace(0, total-1, FRAMES_PER_VIDEO, dtype=int)
        base = os.path.splitext(v)[0]
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            # convert BGR to RGB and save as JPG
            out_name = f"{base}_f{i+1}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, frame)
            count += 1

        cap.release()

    return count

def main():
    print('Extracting frames to', OUT_ROOT)
    real_src = os.path.join(VIDEO_ROOT, 'videos_real')
    fake_src = os.path.join(VIDEO_ROOT, 'videos_fake')

    r = extract_from_dir(real_src, 'real')
    f = extract_from_dir(fake_src, 'fake')

    print(f'Extracted {r} real images and {f} fake images into {OUT_ROOT}')

if __name__ == '__main__':
    main()
