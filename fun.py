import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import os, gc

import torchvision.models as models
from tabulate import tabulate

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def folder_settings(args, folder_path):
    release_gpu_memory()# 释放GPU内存

    # if you have GPU, the device will be cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    # 创建表格数据
    table_data = [
        ["Device", device],
        ["Folder Path", folder_path],
        ["Batch Size", args.batch_size],
        ["Number of Workers", nw],
        ["Learning Rate", args.lr]
    ]
    # 打印表格
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
    return device, nw

