import shutil, os

P='backup'
if not os.path.exists(P):
    print('backup/ not found')
else:
    try:
        shutil.rmtree(P)
        print('backup/ removed')
    except Exception as e:
        print('failed to remove backup:', e)
