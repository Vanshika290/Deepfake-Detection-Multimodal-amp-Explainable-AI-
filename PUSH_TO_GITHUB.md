# How to Push to GitHub

1. The project has been cleaned up. Large files (dataset, checkpoints) are ignored or deleted.
2. The `.gitignore` file ensures only code and necessary assets are tracked.
3. The virtual environment (`venv`) is ignored.

## Steps to Push

1. Open a terminal in this folder.
2. Initialize git if not already done:
   ```bash
   git init
   ```
3. Add files:
   ```bash
   git add .
   ```
   (Note: `dataset/` and large `.pth` files will be automatically skipped due to `.gitignore`)
4. Commit:
   ```bash
   git commit -m "Add deepfake detection project"
   ```
5. Add your remote (if not already added):
   ```bash
   git remote add origin https://github.com/YourUsername/YourRepo.git
   ```
6. Push:
   ```bash
   git push -u origin main
   ```

## Post-Training Cleanup

If you just finished training, you might have `video_model_epoch_X.pth` files.
You can delete them and keep only `video_model_best.pth` and `video_model_final.pth`.
Run this command to clean them:
```python
import glob, os
for f in glob.glob("video_model_epoch_*.pth"):
    try: os.remove(f)
    except: pass
```
