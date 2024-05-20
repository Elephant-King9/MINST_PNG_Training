# convert PNG to Tensor
import torch
from torchvision import transforms


# 将PNG格式的图片转化为张量形式
def PNG_to_Tensor(png_img):
    # 修改图片为单通道
    png_img.convert('L')
    # 设置转化为28x28格式的tensor格式
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    tensor_img = transform(png_img)
    # 转化为1张图1通道28x28格式
    tensor_img = torch.reshape(tensor_img, (1, 1, 28, 28))
    return tensor_img
