"""Train / evaluate your VGGNet on a real dataset.

Supports two dataset modes:
 - cifar10: downloads CIFAR-10 and resizes images to 224x224 (convenient for testing)
 - folder: uses torchvision.datasets.ImageFolder for a dataset in ImageNet-style layout

Usage examples (from repo root):
  python3 Code/CNN_Architecture/train_vgg.py --dataset cifar10 --epochs 3 --batch-size 32
  python3 Code/CNN_Architecture/train_vgg.py --dataset folder --data-dir /path/to/dataset --epochs 10

The script will detect and use MPS on Apple Silicon if available.
"""

import argparse
import pathlib
import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import ssl
import certifi

# Import local VGGNet implementation
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from CNN_Architecture.VGG import VGGNet, VGG_types


def get_device() -> torch.device:
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_data_loaders(dataset_mode: str, data_dir: str, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int]:
    # Use ImageNet normalization for VGG
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
    elif dataset_mode == 'folder':
        train_root = pathlib.Path(data_dir) / 'train'
        val_root = pathlib.Path(data_dir) / 'val'
        if not train_root.exists() or not val_root.exists():
            raise RuntimeError(f"For dataset=folder, expected train/ and val/ subfolders under {data_dir}")
        train_ds = torchvision.datasets.ImageFolder(str(train_root), transform=train_transforms)
        val_ds = torchvision.datasets.ImageFolder(str(val_root), transform=val_transforms)
        num_classes = len(train_ds.classes)
    else:
        raise ValueError('dataset must be one of: cifar10, folder')

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

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'folder'], default='cifar10')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='vgg_checkpoint.pth')
    parser.add_argument('--pretrained', action='store_true', help='Use torchvision pretrained VGG16 (finetune)')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = get_data_loaders(args.dataset, args.data_dir, args.batch_size, args.num_workers)

    if args.pretrained:
        print('Loading torchvision pretrained vgg16 and replacing final classifier')
        vgg = torchvision.models.vgg16(pretrained=True)
        vgg.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        model = vgg.to(device)
    else:
        print('Using local VGGNet implementation (from VGG.py)')
        model = VGGNet(VGG_types['VGG16'], num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        scheduler.step()
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, time: {t1-t0:.1f}s")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.save_path)
            print(f"Saved best checkpoint (acc={best_acc:.4f}) to {args.save_path}")

    total_time = time.time() - start
    print(f"Training finished in {total_time:.1f}s, best_val_acc={best_acc:.4f}")


if __name__ == '__main__':
    main()
