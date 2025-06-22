from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms



def get_mnist_loaders(batch_size=64, num_workers=2, val_ratio=0.1, root='./data'):
    # 定义数据增强和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和方差
    ])

    # 下载并加载训练集
    train_val_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # 划分训练集和验证集
    val_size = int(len(train_val_dataset) * val_ratio)
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 返回类别标签
    target_classes = [str(i) for i in range(10)]

    return train_loader, val_loader, test_loader, target_classes
