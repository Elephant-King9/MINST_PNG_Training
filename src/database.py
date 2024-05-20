# 创建数据集
import os

from PIL import Image
from torch.utils.data import Dataset
from utils import *


class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir, train):
        self.root_dir = root_dir
        self.label_dir = str(label_dir)
        if train:
            self.img_dir = os.path.join(self.root_dir, 'training', self.label_dir)
        else:
            self.img_dir = os.path.join(self.root_dir, 'testing', self.label_dir)
        self.imgs_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs_list[index])
        img = Image.open(img_path)

        # 自定义方法，将PNG格式的图片转化为张量
        tenor_img = PNG_to_Tensor(img)
        label = self.label_dir
        # 返回tensor数据类型和标签
        return tenor_img, label


if __name__ == '__main__':
    root_dir = '../datasets/mnist_png'
    label_dir = 0
    my_dataset = MyDataset(root_dir, label_dir, train=True)
    print(my_dataset)