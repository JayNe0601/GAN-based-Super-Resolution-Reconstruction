import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomCrop, Resize, Normalize
from PIL import Image
class TrainDataset(Dataset):
    def __init__(self, root_dir):
        super(TrainDataset, self).__init__()
        self.root_dir = root_dir
        self.image_filenames = []
        self.transform_hr = RandomCrop(96) # 随机裁剪
        self.transform_lr = Resize(24)
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        for _,_,files in os.walk(self.root_dir):
            for file in files:
                image_file = os.path.join(self.root_dir,file)
                self.image_filenames.append(image_file)


    def __getitem__(self, index):
        hr_filename = self.image_filenames[index]
        hr_image = Image.open(hr_filename).convert('RGB') # 确保图像是RGB格式
        hr_image = self.to_tensor(hr_image) # 转换为[0,1]范围
        hr_image = self.transform_hr(hr_image) # 随机裁剪
        hr_image = self.normalize(hr_image) # 归一化到[-1,1]范围
        hr_image = hr_image.unsqueeze(0)
        lr_image = nn.functional.interpolate(hr_image, scale_factor=1 / 4, mode='bicubic', align_corners=True) # 得到训练的lr图像
        hr_image, lr_image = hr_image.squeeze(0), lr_image.squeeze(0)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    # 使用示例
    dataset = TrainDataset(root_dir=r'E:\Develop\Super_Resolution\Super_Resolution\data\DIV2K_train_HR')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(len(dataloader))
    for lr_images, hr_images in dataloader:
        print(hr_images)