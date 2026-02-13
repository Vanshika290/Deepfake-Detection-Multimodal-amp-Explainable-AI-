import torch
import torch.nn as nn
import torch.optim as optim
import time

print('Starting smoke run to validate image and video training loops')

# Image model smoke test
try:
    from train_image import build_model, DEVICE as IMAGE_DEVICE
    print('\n[Image] Building model...')
    img_model = build_model()
    img_model.train()

    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224, device=IMAGE_DEVICE)
    dummy_labels = torch.randint(0, 2, (batch_size,), device=IMAGE_DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(img_model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    outputs = img_model(dummy_images)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f'[Image] One training step done. Loss={loss.item():.4f}')
    torch.save({'model_state_dict': img_model.state_dict()}, 'smoke_image_ck.pth')
except Exception as e:
    print('[Image] Smoke test failed:', e)

# Video model smoke test
try:
    from train_video import DeepfakeDetector, FRAMES_PER_VIDEO, DEVICE as VIDEO_DEVICE
    print('\n[Video] Building model...')
    vid_model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(VIDEO_DEVICE)
    vid_model.train()

    batch_size = 2
    dummy_frames = torch.randn(batch_size, FRAMES_PER_VIDEO, 3, 224, 224, device=VIDEO_DEVICE)
    dummy_labels = torch.randint(0, 2, (batch_size,), device=VIDEO_DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vid_model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    outputs = vid_model(dummy_frames)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f'[Video] One training step done. Loss={loss.item():.4f}')
    torch.save({'model_state_dict': vid_model.state_dict()}, 'smoke_video_ck.pth')
except Exception as e:
    print('[Video] Smoke test failed:', e)

print('\nSmoke run finished')
