import sys
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import ssl
import certifi

# Make repo root importable so we can import Code/CNN_Architecture/LeNet.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from CNN_Architecture.LeNet import LeNet


def get_data_loaders(batch_size=64, data_dir='./data'):
    # MNIST is 28x28; LeNet implementation expects 32x32.
    # We'll resize to 32x32 here. Alternatively you can pad to 32x32 or change the model.
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Workaround for macOS/venv SSL certificate issues: set ssl context using certifi bundle
    try:
        # assign a callable that returns an SSLContext configured with certifi CA bundle
        ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # If certifi isn't available, torchvision will try to download and may fail with CERTIFICATE_VERIFY_FAILED.
        # If you see that error, install certifi into your venv: pip install certifi
        pass

    try:
        train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    except Exception as e:
        # Provide a helpful error message to the user
        raise RuntimeError(
            "Failed to download MNIST. If you're on macOS with a virtual environment, try: `pip install certifi` "
            "and re-run. Original error: " + str(e)
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += imgs.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += imgs.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main(
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = None,
    save_path: str = 'lenet_mnist.pth',
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    model = LeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on device: {device}, epochs={epochs}, batch_size={batch_size}")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        t1 = time.time()
        print(
            f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, time: {t1-t0:.1f}s"
        )

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s")
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == '__main__':
    main()
