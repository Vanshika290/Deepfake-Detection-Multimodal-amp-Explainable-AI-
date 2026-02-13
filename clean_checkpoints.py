import glob
import os

print("Deleting all intermediate epoch checkpoints...")
files = glob.glob("video_model_epoch_*.pth")
count = 0
for f in files:
    try:
        os.remove(f)
        print(f"Deleted {f}")
        count += 1
    except Exception as e:
        print(f"Error deleting {f}: {e}")

print(f"\nDeleted {count} files.")
print("Kept 'video_model_best.pth' and 'video_model_final.pth'!")
