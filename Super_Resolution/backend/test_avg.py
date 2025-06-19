import torch
import torchvision.transforms as transforms
from PIL import Image
from models import Generator
import math
import numpy as np
import os
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.features.eval()

        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 提取VGG16的中间层特征
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {3, 8, 15, 22}:  # 选择特定的层
                features.append(x)
        return features


def calculate_lpips(img1, img2, device):
    """计算LPIPS (Learned Perceptual Image Patch Similarity)"""
    # 初始化特征提取器
    feature_extractor = VGGFeatureExtractor().to(device)
    feature_extractor.eval()

    with torch.no_grad():
        # 提取特征
        features1 = feature_extractor(img1)
        features2 = feature_extractor(img2)

        # 计算每个层的特征差异
        lpips_value = 0
        for f1, f2 in zip(features1, features2):
            # 计算特征图之间的欧氏距离
            diff = (f1 - f2) ** 2
            lpips_value += torch.mean(diff)

        # 归一化
        lpips_value = lpips_value / len(features1)

    return lpips_value.item()


def tensor_to_numpy(tensor):
    """将tensor转换为numpy数组"""
    # 将[-1,1]范围的tensor转换回[0,1]范围
    tensor = (tensor + 1) / 2.0
    # 确保值在[0,1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().numpy().transpose(1, 2, 0)


def calculate_mse(img1, img2):
    """计算均方误差"""
    return np.mean((img1 - img2) ** 2)


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel) - 10 * math.log10(mse)
    return psnr


def calculate_ssim(img1, img2, k1=0.01, k2=0.03, L=1.0):
    """计算SSIM"""
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # 计算均值
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # 计算方差和协方差
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    # 计算SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    return ssim


def test_avg(lr_data_dir, hr_data_dir, save_dir, generator_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载预训练的SRGAN模型
    model = Generator()
    model_dict = torch.load(generator_dir)
    model.load_state_dict(model_dict)
    model.eval()
    model = model.to(device)
    print("生成器模型加载完毕")

    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 获取所有测试图像
    lr_images = sorted([f for f in os.listdir(lr_data_dir) if f.endswith('.png')])
    hr_images = sorted([f for f in os.listdir(hr_data_dir) if f.endswith('.png')])

    # 初始化指标列表
    psnr_list = []
    ssim_list = []
    lpips_list = []

    # 遍历所有图像
    for lr_name, hr_name in tqdm(zip(lr_images, hr_images), total=len(lr_images), desc="处理测试图像"):
        # 加载图像
        lr_path = os.path.join(lr_data_dir, lr_name)
        hr_path = os.path.join(hr_data_dir, hr_name)

        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')

        # 对图像进行预处理
        lr_tensor = transform(lr_img).unsqueeze(0).to(device)
        hr_tensor = transform(hr_img).unsqueeze(0).to(device)

        # 生成超分辨率图像
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # 转换为numpy数组
        hr_np = tensor_to_numpy(hr_tensor[0])
        sr_np = tensor_to_numpy(sr_tensor[0])

        # 计算指标
        psnr = calculate_psnr(hr_np, sr_np)
        ssim = calculate_ssim(hr_np, sr_np)

        # 计算LPIPS
        hr_tensor_lpips = (hr_tensor + 1) / 2
        sr_tensor_lpips = (sr_tensor + 1) / 2
        lpips_value = calculate_lpips(hr_tensor_lpips, sr_tensor_lpips, device)

        # 添加到列表
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_value)

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算平均值
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)

    # 计算标准差
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)
    std_lpips = np.std(lpips_list)

    # 保存结果
    results = {
        'PSNR': {'mean': avg_psnr, 'std': std_psnr},
        'SSIM': {'mean': avg_ssim, 'std': std_ssim},
        'LPIPS': {'mean': avg_lpips, 'std': std_lpips}
    }

    # 打印结果
    print("\n测试集评估结果:")
    print(f"PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}")

    # 保存详细结果到文件
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write("测试集评估结果:\n")
        f.write(f"PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}\n\n")

        f.write("每张图片的详细结果:\n")
        for i, (lr_name, hr_name) in enumerate(zip(lr_images, hr_images)):
            f.write(f"\n图片 {i + 1}: {lr_name}\n")
            f.write(f"PSNR: {psnr_list[i]:.2f} dB\n")
            f.write(f"SSIM: {ssim_list[i]:.4f}\n")
            f.write(f"LPIPS: {lpips_list[i]:.4f}\n")

if __name__ == '__main__':
    # 测试集路径
    lr_data_dir = r"E:\Develop\Super_Resolution\Super_Resolution\data\DIV2K_test_LR_bicubic\X4"
    hr_data_dir = r"E:\Develop\Super_Resolution\Super_Resolution\data\DIV2K_test_HR"
    save_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\RUN"
    generator_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\SRGAN\256_5e-5\generator_100.pth"

    test_avg(lr_data_dir, hr_data_dir, save_dir, generator_dir)
