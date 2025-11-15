# utils/dataloader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

# 0 workers to avoid pickling issues on Mac/Windows
NUM_WORKERS = 0

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def _test_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

def _tta_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

def get_cifar10_loader(batch_size=128, img_size=224, tta=False):
    tfm = _tta_transform(img_size) if tta else _test_transform(img_size)
    ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

def get_cifar100_loader(batch_size=128, img_size=224, tta=False):
    tfm = _tta_transform(img_size) if tta else _test_transform(img_size)
    ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

def get_svhn_loader(batch_size=128, img_size=224, tta=False):
    # SVHN images are already PIL or ndarray depending on version – no ToPILImage needed
    tfm = _tta_transform(img_size) if tta else _test_transform(img_size)
    ds = datasets.SVHN(root='./data', split='test', download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

def get_tinyimagenet_loader(batch_size=128, img_size=224, tta=False):
    tfm = _tta_transform(img_size) if tta else _test_transform(img_size)

    root = "/content/tiny-imagenet-200/val_fixed"

    ds = datasets.ImageFolder(root=root, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

