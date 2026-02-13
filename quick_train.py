"""Quick training runner: runs 1 epoch for image and video training to validate end-to-end.
This is a short run — it reduces epoch counts and batch sizes where possible.
"""
import time

print('Starting quick training: 1 epoch for image and video models')

try:
    import train_image
    print('\nRunning image training (1 epoch)...')
    train_image.NUM_EPOCHS = 1
    train_image.BATCH_SIZE = 16
    start = time.time()
    train_image.train()
    print(f'Image quick train done in {time.time()-start:.1f}s')
except Exception as e:
    print('Image training failed or skipped:', e)

try:
    import train_video
    print('\nRunning video training (1 epoch)...')
    train_video.NUM_EPOCHS = 1
    train_video.BATCH_SIZE = 4
    start = time.time()
    train_video.main()
    print(f'Video quick train done in {time.time()-start:.1f}s')
except Exception as e:
    print('Video training failed or skipped:', e)

print('\nQuick training finished')
