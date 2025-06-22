from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=64, num_workers=2, val_ratio=0.1, root='./data'):
    # CIFAR-10 均值和方差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 下载并加载训练集
    train_val_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    # 划分训练集和验证集
    val_size = int(len(train_val_dataset) * val_ratio)
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # CIFAR-10 类别标签
    target_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']

    return train_loader, val_loader, test_loader, target_classes