import os
import shutil
import glob
import sys
import time

# Redirect stdout to a log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training_log.txt", "w", buffering=1, encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except Exception:
            pass

sys.stdout = Logger()

def cleanup():
    print("Cleaning up unnecessary files...")
    
    # Remove __pycache__ folders recursively but exclude dataset and venv
    exclude_dirs = {"dataset", "venv", ".git", "SDFVD Small-scale Deepfake Forgery Video Dataset"}
    
    for root, dirs, files in os.walk(".", topdown=True):
        # Modify dirs in-place to exclude certain directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for name in dirs:
            if name == "__pycache__":
                try:
                    full_path = os.path.join(root, name)
                    shutil.rmtree(full_path)
                    print(f"Deleted {full_path}")
                except Exception as e:
                    print(f"Skipping {name}: {e}")

    # Remove removed_large
    if os.path.exists("removed_large"):
        try:
           shutil.rmtree("removed_large")
           print("Deleted removed_large")
        except Exception as e:
           print(f"Could not delete removed_large: {e}")

    # Remove old checkpoints
    # We want to keep ONLY the final result of THIS run.
    # But since we haven't run yet, we should delete old ones.
    for pth in glob.glob("*.pth"):
        if "video_model_best.pth" not in pth and "video_model_final.pth" not in pth:
             try:
                 os.remove(pth)
                 print(f"Deleted {pth}")
             except Exception as e:
                 print(f"Error deleting {pth}: {e}")

def run_training():
    print("\nStarting video training...")
    try:
        # Import here to avoid locking __pycache__ before cleanup
        import train_video
        
        # Override settings for a balanced run
        train_video.NUM_EPOCHS = 10 
        train_video.BATCH_SIZE = 8 # Safe for CPU
        train_video.main()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cleanup()
    run_training()
