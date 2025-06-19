import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from models import Generator
import math
import numpy as np
import os
import torch.nn as nn
import torchvision.models as models

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

def tensor_to_numpy(tensor):
    """将tensor转换为numpy数组，并进行反归一化"""
    # 将[-1,1]范围的tensor转换回[0,1]范围
    tensor = (tensor + 1) / 2.0
    # 确保值在[0,1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().numpy().transpose(1, 2, 0)

def normalize_image(img):
    """将图像归一化到[0,1]范围"""
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)

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


def add_text_to_image(image, text, font_size=30):
    """在图片上方添加文字"""
    # 创建一个新的图片，高度比原图多出文字区域
    lines = text.split('\n')
    line_height = 35  # 每行文字的高度
    text_area_height = line_height * len(lines) + 20  # 文字区域总高度
    new_height = image.height + text_area_height
    new_image = Image.new('RGB', (image.width, new_height), (255, 255, 255))

    # 粘贴原图
    new_image.paste(image, (0, text_area_height))

    # 添加文字
    draw = ImageDraw.Draw(new_image)
    try:
        # 尝试加载系统字体
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # 如果找不到系统字体，使用默认字体
        font = ImageFont.load_default()

    # 逐行绘制文字
    for i, line in enumerate(lines):
        # 计算每行文字的位置使其居中
        text_width = draw.textlength(line, font=font)
        text_position = ((image.width - text_width) // 2, 10 + i * line_height)
        # 绘制文字
        draw.text(text_position, line, fill=(0, 0, 0), font=font)

    return new_image


def create_comparison_image(lr_img, hr_img, sr_img, psnr_value, ssim_value):
    """创建包含三张图片的对比图"""
    # 确保所有图片大小一致
    width = hr_img.width
    height = hr_img.height

    # 调整低分辨率图片大小
    lr_img = lr_img.resize((width, height), Image.LANCZOS)

    # 创建新的画布，增加顶部空间用于显示文字
    text_height = 80  # 为文字预留的空间
    comparison_img = Image.new('RGB', (width * 3, height + text_height), (255, 255, 255))

    # 粘贴三张图片
    comparison_img.paste(lr_img, (0, text_height))
    comparison_img.paste(hr_img, (width, text_height))
    comparison_img.paste(sr_img, (width * 2, text_height))

    # 添加文字信息
    draw = ImageDraw.Draw(comparison_img)
    try:
        font = ImageFont.truetype("arial.ttf", 45)
    except:
        font = ImageFont.load_default()

    # 添加每张图片的标题
    draw.text((width // 2 - 70, 20), "LR", fill=(0, 0, 0), font=font)
    draw.text((width + width // 2 - 70, 20), "HR", fill=(0, 0, 0), font=font)

    # 在SR图片上方添加PSNR和SSIM信息
    metrics_text = f"PSNR: {psnr_value:.2f} dB  SSIM: {ssim_value:.4f}"
    text_width = draw.textlength(metrics_text, font=font)
    draw.text((width*2 + (width - text_width)//2, 20), metrics_text, fill=(0, 0, 0), font=font)

    return comparison_img


def test(lr_data_dir, hr_data_dir, save_dir, generator_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
        将一张图片加载进来，并得到超分辨率后的图像
    '''

    # 加载预训练的SRGAN模型
    model = Generator()
    model_dict = torch.load(generator_dir)
    model.load_state_dict(model_dict)
    model.eval()
    model = model.to(device)
    print(f"生成器模型加载完毕，使用设备: {device}")

    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载测试图像
    lr_img = Image.open(lr_data_dir).convert('RGB')
    hr_img = Image.open(hr_data_dir).convert('RGB')

    # 对测试图像进行预处理
    lr_tensor = transform(lr_img).unsqueeze(0).to(device)
    hr_tensor = transform(hr_img).unsqueeze(0).to(device)

    # 将测试图像输入SRGAN模型进行超分辨率重建
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 将tensor转换为numpy数组用于计算指标
    lr_np = tensor_to_numpy(lr_tensor[0])
    hr_np = tensor_to_numpy(hr_tensor[0])
    sr_np = tensor_to_numpy(sr_tensor[0])

    # 计算PSNR和SSIM
    psnr_value = calculate_psnr(hr_np, sr_np)
    ssim_value = calculate_ssim(hr_np, sr_np)

    # 计算LPIPS
    # 将图像转换到[0,1]范围用于LPIPS计算
    hr_tensor_lpips = (hr_tensor + 1) / 2
    sr_tensor_lpips = (sr_tensor + 1) / 2
    lpips_value = calculate_lpips(hr_tensor_lpips, sr_tensor_lpips, device)

    # 创建对比图像
    # 将图像转换回PIL格式
    sr_img = sr_tensor.squeeze(0).cpu().detach()
    sr_img = (sr_img + 1) / 2.0  # 反归一化到[0,1]范围
    sr_img = torch.clamp(sr_img, 0, 1)  # 确保值在[0,1]范围内
    sr_img = (sr_img * 255.0).to(torch.uint8)  # 转换到[0,255]范围
    sr_img = sr_img.numpy().transpose(1, 2, 0)
    sr_img = Image.fromarray(sr_img)

    # 创建对比图像
    comparison_img = create_comparison_image(lr_img, hr_img, sr_img, psnr_value, ssim_value)

    # 保存结果
    sr_save_path = os.path.join(save_dir, 'sr_result.png')
    comparison_save_path = os.path.join(save_dir, 'comparison_result.png')
    sr_img.save(sr_save_path)
    comparison_img.save(comparison_save_path)
    print(f"超分辨率图像已保存到: {sr_save_path}")
    print(f"对比图已保存到: {comparison_save_path}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"LPIPS: {lpips_value:.4f}")

def test2(p_dir, save_dir, generator_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
        将一张图片加载进来，并得到超分辨率后的图像
    '''

    # 加载预训练的SRGAN模型
    model = Generator()
    model_dict = torch.load(generator_dir)
    model.load_state_dict(model_dict)
    model.eval()
    model = model.to(device)
    print(f"生成器模型加载完毕，使用设备: {device}")

    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载测试图像
    img = Image.open(p_dir).convert('RGB')

    # 对测试图像进行预处理
    tensor = transform(img).unsqueeze(0).to(device)

    # 将测试图像输入SRGAN模型进行超分辨率重建
    with torch.no_grad():
        sr_tensor = model(tensor)

    # 创建对比图像
    # 将图像转换回PIL格式
    sr_img = sr_tensor.squeeze(0).cpu().detach()
    sr_img = (sr_img + 1) / 2.0  # 反归一化到[0,1]范围
    sr_img = torch.clamp(sr_img, 0, 1)  # 确保值在[0,1]范围内
    sr_img = (sr_img * 255.0).to(torch.uint8)  # 转换到[0,255]范围
    sr_img = sr_img.numpy().transpose(1, 2, 0)
    sr_img = Image.fromarray(sr_img)

    # 保存对比结果
    sr_save_path = os.path.join(save_dir, 'input6_sr.png')
    sr_img.save(sr_save_path)
    print(f"超分辨率图像已保存到: {sr_save_path}")

if __name__ == '__main__':
    # # 0896x4.png，0843x4.png, 0844x4.png. 0802X4.png, 0803x4.png, 0838x4.png, 0857x4.png, 0862x4.png, 0877x4.png
    lr_data_dir = r"E:\Develop\Super_Resolution\Super_Resolution\data\DIV2K_test_LR_bicubic\X4\0807x4.png"
    hr_data_dir = r"E:\Develop\Super_Resolution\Super_Resolution\data\DIV2K_test_HR\0807.png"
    save_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\RUN"
    generator_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\SRGAN\256_5e-5\generator_100.pth"
    test(lr_data_dir, hr_data_dir, save_dir, generator_dir)
    # p_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\RUN\test\input6.png"
    # save_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\RUN\test"
    # generator_dir = r"/result/SRGAN/96_1e-4\generator_100.pth"
    # test2(p_dir, save_dir, generator_dir)