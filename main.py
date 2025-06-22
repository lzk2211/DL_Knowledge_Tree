import sys
# sys.path.append('/home/lab125/kk/MIML_DCNN') # please change to your path

import os

import torch
import torch.nn as nn

from tqdm import tqdm

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps  # 注意替代 cm
import json

from dataset.MNIST import get_mnist_loaders
from dataset.Cifar import get_cifar10_loaders

from fun import folder_settings
from train_val_test import *
from model import ResNet
from model import LeNet
from model import densenet
from model import MLP
import torchvision.models as models

def main():
    parser = argparse.ArgumentParser(description="Train a localization model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--Train', type=bool, default=True, help='Switch wheater to train')
    parser.add_argument('--Test', type=bool, default=True, help='Test the test_loader use trained model')
    parser.add_argument('--PCA', type=bool, default=True, help='The PCA Visualisation')
    parser.add_argument('--tSNE', type=bool, default=True, help='The t-SNE Visualisation')
    parser.add_argument('--UMAP', type=bool, default=True, help='The UMAP Visualisation')
    parser.add_argument('--exp_name', type=str, default='MNIST_LeNet_Classification', help='the name of the experiment')

    args = parser.parse_args()
    print(args)


    folder_path = './results/' + args.exp_name
    if not os.path.exists(folder_path):# if the folder does not exist, than create it
        os.makedirs(folder_path)


    device, nw = folder_settings(args, folder_path)

    train_loader, val_loader, test_loader, target_classes = get_mnist_loaders(batch_size=args.batch_size, num_workers=nw, val_ratio=0.1, root='./data')
    # train_loader, val_loader, test_loader, target_classes = get_cifar10_loaders(batch_size=args.batch_size, num_workers=nw, val_ratio=0.1, root='./data')

    # Update the model to fit your input size and number of classes
    # model = ResNet.resnet34(num_classes=10, include_top=True)
    # model = models.resnet34(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 10)

    model = LeNet.LeNet()
    # model = LeNet.Simple_LeNet()

    # model = densenet.densenet121(num_classes=10)
    # model = MLP.SimpleMLP(num_classes=10)

    model.to(device)

    if args.Train:
        train(args, folder_path, model, device, train_loader, val_loader)


    if args.Test:
        test(args, folder_path, model, device, test_loader, target_classes)


if __name__ == '__main__':
    main()
