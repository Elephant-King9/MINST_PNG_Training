import os.path

import torch
import torchvision
from PIL import Image

root_dir = '../datasets/mnist_png'
label_dir = 0
# print(label_dir[2])

path = os.path.join(root_dir, 'training', str(label_dir))
print(path)
imgs_list = os.listdir(path)

img_path = os.path.join(path, imgs_list[0])
img = Image.open(img_path)
print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                            torchvision.transforms.ToTensor()])
img_tensor = transform(img)
print(type(img_tensor))
img_tensor = torch.reshape(img_tensor, (1, 1, 28, 28))
print(img_tensor.shape)
