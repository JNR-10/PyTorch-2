import sys
import pathlib
import time
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import ssl
import certifi

# Make repo root importable so we can import Code/CNN_Architecture/InceptionNet.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from CNN_Architecture.InceptionNet import Inception_block, conv_block

def get_device() -> torch.device:
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_data_loaders(dataset_mode: str, data_dir: str, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    # Use ImageNet normalization for InceptionNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_mode == 'cifar10':
        # Workaround for macOS/venv SSL certificate issues: configure default HTTPS context to use certifi bundle
        try:
            ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
        except Exception:
            # If certifi isn't installed or something goes wrong, we'll try the download and show a helpful error
            pass

        try:
            train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
            val_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transforms)
        except Exception as e:
            raise RuntimeError(
                "Failed to download CIFAR-10. On macOS inside a virtualenv you may need to: `pip install certifi` "
                "or run the OS Python 'Install Certificates.command'. Original error: " + str(e)
            )
        num_classes = 10

    elif dataset_mode == 'imagenet':
        try:
            ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
        try:
            train_ds = torchvision.datasets.ImageNet(root=data_dir, split='train', download=True, transform=train_transforms)
            val_ds = torchvision.datasets.ImageNet(root=data_dir, split='val', download=True, transform=val_transforms)
        except Exception as e:
            raise RuntimeError(
                "Failed to download ImageNet. On macOS inside a virtualenv you may need to: `pip install certifi` "
                "or run the OS Python 'Install Certificates.command'. Original error: " + str(e)
            )
        num_classes = 1000
    elif dataset_mode == 'custom':
        train_root = pathlib.Path(data_dir) / 'train'
        val_root = pathlib.Path(data_dir) / 'val'
        if not train_root.exists() or not val_root.exists():
            raise RuntimeError(f"For dataset=custom, expected train/ and val/ subfolders under {data_dir}")
        train_ds = torchvision.datasets.ImageFolder(str(train_root), transform=train_transforms)
        val_ds = torchvision.datasets.ImageFolder(str(val_root), transform=val_transforms)
        num_classes = len(train_ds.classes)
    else:
        raise ValueError(f"Unsupported dataset mode: {dataset_mode}")   


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, num_classes

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

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'imagenet', 'custom'], default='cifar10')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    train_loader, val_loader, num_classes = get_data_loaders(
        dataset_mode=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = Inception_block(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        epoch_time = time.time() - start_time

        print(f'Epoch [{epoch+1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | '
              f'Time: {epoch_time:.2f}s')
        
if __name__ == '__main__':
    main()