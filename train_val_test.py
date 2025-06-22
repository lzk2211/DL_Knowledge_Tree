import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
from confusionMatrix import ConfusionMatrix  # 添加此行
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from umap import UMAP
import sys, select

def plot_umap(features, labels, epoch, folder_path, num_classes=10):
    reducer = UMAP(n_components=2, random_state=0)
    features_2d = reducer.fit_transform(features)

    umap_dir = os.path.join(folder_path, "umap")
    if not os.path.exists(umap_dir):
        os.makedirs(umap_dir)

    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        idxs = labels == i
        plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=str(i), s=5)
    plt.legend(loc='upper right')  # 强制图例在右上
    plt.title(f"UMAP at Epoch {epoch+1}")
    plt.savefig(os.path.join(umap_dir, f"umap_epoch_{epoch+1}.png"))
    plt.close()

def plot_tsne(features, labels, epoch, folder_path, num_classes=10):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    tsne_dir = os.path.join(folder_path, "tsne")
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir)

    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        idxs = labels == i
        plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=str(i), s=5)
    plt.legend(loc='upper right')  # 强制图例在右上
    plt.title(f"t-SNE at Epoch {epoch+1}")
    plt.savefig(os.path.join(tsne_dir, f"tsne_epoch_{epoch+1}.png"))
    plt.close()


def plot_pca(features, labels, epoch, folder_path, num_classes=10):
    pca = PCA(n_components=2, random_state=0)
    features_2d = pca.fit_transform(features)
    # 创建pca文件夹
    pca_dir = os.path.join(folder_path, "pca")
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)

    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        idxs = labels == i
        plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=str(i), s=5)
    plt.legend(loc='upper right')  # 强制图例在右上
    plt.title(f"PCA at Epoch {epoch+1}")
    plt.savefig(os.path.join(pca_dir, f"pca_epoch_{epoch+1}.png"))
    plt.close()

def train(args, folder_path, model, device, train_loader, val_loader):
    if sys.platform.startswith('linux'):
        system=0
    elif sys.platform.startswith('win'):
        system=1
        import msvcrt

    writer = SummaryWriter(os.path.join(folder_path, 'runs'))
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_acc = 0.0
    save_path = os.path.join(folder_path, 'model.pth')
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        features_list = []
        labels_list = []
        train_bar = tqdm(train_loader, file=None)
        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, feats = model(data, feature=True)
            feats_flat = feats.view(feats.size(0), -1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_bar.set_description(f"Train Epoch [{epoch+1}/{args.epochs}] Loss: {loss.item():.3f}")

            features_list.append(feats_flat.detach().cpu())
            labels_list.append(labels.detach().cpu())

        train_acc = correct / total
        train_loss /= len(train_loader)

        # 可视化
        feats = torch.cat(features_list, dim=0)
        feats = feats.view(feats.size(0), -1).numpy()  # 展平为 [N, D]
        labs = torch.cat(labels_list, dim=0).numpy()

        if hasattr(args, "PCA") and args.PCA:
            plot_pca(feats, labs, epoch, folder_path, num_classes=10)

        if hasattr(args, "UMAP") and args.UMAP:
            plot_umap(feats, labs, epoch, folder_path, num_classes=10)

        if hasattr(args, "tSNE") and args.tSNE:
            plot_tsne(feats, labs, epoch, folder_path, num_classes=10)

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs, _ = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_loss /= len(val_loader)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if system == 1 and msvcrt.kbhit():  # 检查是否有键盘输入
            user_input = msvcrt.getch().decode('utf-8').strip()
            if user_input == 'q':
                print("Stopping training after this epoch.")
                break
        elif system == 0 and sys.stdin in select.select([sys.stdin], [], [], 0)[0]:  # Check if 'q' was pressed
            user_input = sys.stdin.readline().strip()
            if user_input == 'q':
                print("Stopping training after this epoch.")
                break



    print('Finished Training')
    print(f'Training time: {time.time() - start_time:.2f} seconds')
    writer.close()

def test(args, folder_path, model, device, test_loader, target_classes):
    print('Testing')
    model_path = os.path.join(folder_path, 'model.pth')
    assert os.path.exists(model_path), f"cannot find {model_path} file"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    confusion = ConfusionMatrix(num_classes=len(target_classes), labels=target_classes)

    with torch.no_grad():
        for data, targets in tqdm(test_loader, file=None):
            data, targets = data.to(device), targets.to(device)
            outputs, _ = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            confusion.update(predicted.cpu().numpy(), targets.cpu().numpy())

    acc = correct / total
    print(f'[Test] Accuracy: {acc:.3f}')
    confusion.plot(folder_path)
    confusion.summary()
    return acc
