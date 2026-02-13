import os
import shutil

ROOT = os.path.abspath(os.path.dirname(__file__))
BACKUP_DIR = os.path.join(ROOT, 'backup')

PATTERNS_TO_MOVE = [
    'deepfake_model_epoch_*.pth',
    'video_model_epoch_*.pth',
    'deepfake_model_fast.pth',
    'test_results*.txt',
    'quick_start*',
    'QUICK_START.md',
]

# Files to KEEP in place (do not move)
KEEP = {
    'deepfake_model_final.pth',
    'video_model_best.pth',
    'video_model_final.pth',
    'smoke_image_ck.pth',
    'smoke_video_ck.pth',
}

def find_files(root, pattern):
    import fnmatch
    matches = []
    for f in os.listdir(root):
        if fnmatch.fnmatch(f, pattern):
            matches.append(os.path.join(root, f))
    return matches

def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    moved = []
    candidates = set()

    for pat in PATTERNS_TO_MOVE:
        found = find_files(ROOT, pat)
        for p in found:
            candidates.add(p)

    # Filter out keep files
    to_move = [p for p in candidates if os.path.basename(p) not in KEEP]

    if not to_move:
        print('No candidate files found to move.')
        return

    print('Files to be moved to backup:')
    for p in to_move:
        print('  -', os.path.basename(p))

    # Move
    for p in to_move:
        try:
            dest = os.path.join(BACKUP_DIR, os.path.basename(p))
            shutil.move(p, dest)
            moved.append(dest)
        except Exception as e:
            print(f'Failed to move {p}: {e}')

    print('\nMoved files:')
    for m in moved:
        print('  -', os.path.relpath(m, ROOT))

    print(f'Backup completed. {len(moved)} files moved to {os.path.relpath(BACKUP_DIR, ROOT)}')

if __name__ == '__main__':
    main()
