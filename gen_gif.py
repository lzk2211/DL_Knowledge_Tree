from PIL import Image
import glob
import os

# 获取所有图片文件名，并按修改时间排序
img_files = sorted(
    glob.glob('./results/MNIST_LeNet_Classification/umap/*.png'),
    key=os.path.getmtime
)

# 打开所有图片
imgs = [Image.open(f) for f in img_files]

# 保存为gif
imgs[0].save('output.gif', save_all=True, append_images=imgs[1:], duration=300, loop=0)

