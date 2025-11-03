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

# Make repo root importable so we can import Code/CNN_Architecture/VGG.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from CNN_Architecture.EfficientNet import EfficientNet

def get_device() -> torch.device:
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_data_loaders(dataset_mode: str, data_dir: str, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    # Use ImageNet normalization for EfficientNet
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
        # Workaround for macOS/venv SSL certificate issues: configure default HTTPS context to use certifi bundle
        try:
            ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
        except Exception:
            # If certifi isn't installed or something goes wrong, we'll try the download and show a helpful error
            pass

        try:
            train_ds = torchvision.datasets.ImageNet(root=data_dir, split='train', download=False, transform=train_transforms)
            val_ds = torchvision.datasets.ImageNet(root=data_dir, split='val', download=False, transform=val_transforms)
        except Exception as e:
            raise RuntimeError(
                "Failed to load ImageNet. On macOS inside a virtualenv you may need to: `pip install certifi` "
                "or run the OS Python 'Install Certificates.command'. Original error: " + str(e)
            )
        num_classes = 1000

    else:
        raise ValueError('dataset must be one of: cifar10, imagenet')
    
    train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_data_loader, val_data_loader

def train_one_epoch(model: nn.Module, device: torch.device, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
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

def evaluate(model: nn.Module, device: torch.device, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
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
    parser = argparse.ArgumentParser(description='Train EfficientNet on CIFAR-10 or ImageNet')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'], required=True, help='Dataset to use for training')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory to download/load the dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads for data loading')
    args = parser.parse_args()

    device = get_device()
    train_loader, val_loader = get_data_loaders(args.dataset, args.data_dir, args.batch_size, args.num_workers)
    num_classes = 10 if args.dataset == 'cifar10' else 1000

    model = EfficientNet(version='b0', num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch+1}/{args.epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - '
              f'Time: {epoch_time:.2f}s')
        
if __name__ == '__main__':
    main()