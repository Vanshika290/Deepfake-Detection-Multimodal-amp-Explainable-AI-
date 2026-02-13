import os

video_root = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD"
fake_dir = os.path.join(video_root, 'videos_fake')
real_dir = os.path.join(video_root, 'videos_real')

if os.path.exists(fake_dir):
    fake_videos = [f for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    print(f"Fake videos: {len(fake_videos)}")
    print(f"Sample fake: {fake_videos[:3]}")
else:
    print(f"Fake directory not found: {fake_dir}")

if os.path.exists(real_dir):
    real_videos = [f for f in os.listdir(real_dir) if f.endswith('.mp4')]
    print(f"Real videos: {len(real_videos)}")
    print(f"Sample real: {real_videos[:3]}")
else:
    print(f"Real directory not found: {real_dir}")
