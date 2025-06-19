import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models import Generator, Discriminator
from train_dataloader import TrainDataset
import os
import matplotlib.pyplot as plt
import logging
import torchvision.models as models
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel) - 10 * math.log10(mse)
    return psnr

def plot_losses(losses_dict, save_dir):
    """绘制损失曲线并保存"""
    ensure_dir(save_dir)
    plt.figure(figsize=(15, 10))
    for name, values in losses_dict.items():
        plt.plot(values, label=name)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

def plot_individual_losses(losses_dict, save_dir):
    """为每个损失单独绘制曲线"""
    ensure_dir(save_dir)
    for name, values in losses_dict.items():
        plt.figure(figsize=(10, 6))
        plt.plot(values, label=name)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title(f'{name}损失曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{name}_curve.png'))
        plt.close()

def plot_psnr(psnr_values, save_dir):
    """绘制PSNR随轮次变化的曲线"""
    ensure_dir(save_dir)
    plt.figure(figsize=(10, 6))
    plt.plot(psnr_values, label='PSNR')
    plt.xlabel('训练轮次')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR随训练轮次的变化')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'psnr_curve.png'))
    plt.close()

def train(batch_size, lr, num_epochs, scale_factor, root_dir):
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = batch_size # 批量大小
    lr = lr # 学习率
    num_epochs = num_epochs # 训练轮数
    scale_factor = scale_factor # 超分辨率缩放因子
    root_dir = root_dir # 'data/DIV2K_train_HR'

    # 初始化损失记录列表
    losses = {
        '生成器损失': [],
        '判别器损失': [],
        '真实图像损失': [],
        '生成图像损失': [],
        '内容损失': [],
        '对抗损失': []
    }

    # 初始化PSNR记录列表
    psnr_values = []

    # 加载数据集
    train_dataset = TrainDataset(root_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 增加数据加载的工作进程数
    )
    print(f"数据集加载完毕，共{len(train_dataset)}张图像")

    # 定义生成器和判别器模型
    generator = Generator(scale_factor=scale_factor).to(device)
    discriminator = Discriminator().to(device)
    pth_path = r'E:\Develop\Super_Resolution\Super_Resolution\data\model\generator_100.pth'
    generator_dict = torch.load(pth_path)
    generator.load_state_dict(generator_dict)
    print(f"生成器权重加载完毕")

    # 定义损失函数和优化器
    content_loss = nn.MSELoss().to(device)  # 均方误差
    adversarial_loss = nn.BCELoss().to(device)  # 二分类交叉熵
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # # 加载VGG19模型
    # vgg19 = models.vgg19(pretrained=True).features
    # # 冻结VGG19模型的权重
    # for param in vgg19.parameters():
    #     param.requires_grad = False
    # # 定义VGG19模型的损失函数
    # content_loss_vgg = nn.MSELoss()
    # if torch.cuda.is_available():
    #     vgg19.to(device)
    #     content_loss_vgg.to(device)
    # logger.info(f"VGG权重加载完毕")

    # 确保保存路径存在
    save_path = r'E:\Develop\Super_Resolution\Super_Resolution\result\SRGAN\96_5e-5_ep_200'
    ensure_dir(save_path)

    print("开始训练模型...")
    # 训练模型
    for epoch in range(num_epochs):
        generator_loss_total = 0
        discriminator_loss_total = 0
        real_loss_total = 0
        fake_loss_total = 0
        content_loss_value_total = 0
        adversarial_loss_value_total = 0
        epoch_psnr = 0

        for i, (lr_images, hr_images) in enumerate(train_loader):
            # 将数据移动到GPU上
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            # 计算生成器损失
            generated_high_res = generator(lr_images)  # 生成器生成
            content_loss_value = content_loss(hr_images, generated_high_res)  # 均方误差 高分辨率 vs 生成高分辨率
            content_loss_value_total += content_loss_value.item()
            adversarial_loss_value = adversarial_loss(discriminator(generated_high_res), torch.ones(batch_size, 1).to(device))  # 二分类交叉熵
            adversarial_loss_value_total += adversarial_loss_value.item()
            generator_loss = content_loss_value + 1e-3 * adversarial_loss_value

            # # 计算VGG19损失
            # high_res_vgg = vgg19(hr_images)
            # generated_high_res_vgg = vgg19(generated_high_res)
            # content_loss_vgg_value = content_loss_vgg(high_res_vgg, generated_high_res_vgg)
            # generator_loss += content_loss_vgg_value
            # generator_loss /= 2

            generator_loss_total += generator_loss.item()

            # 更新生成器权重
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            # 计算判别器损失
            dis_hr = discriminator(hr_images)
            dis_generated = discriminator(generated_high_res.detach()).detach()

            # 非软标签版
            real_loss = adversarial_loss(discriminator(hr_images), torch.ones(batch_size, 1).to(device)) # 真实图像的损失
            real_loss_total += real_loss.item()
            fake_loss = adversarial_loss(discriminator(generated_high_res.detach()), torch.zeros(batch_size, 1).to(device)) # 生成图像的损失
            fake_loss_total += fake_loss.item()
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss_total += discriminator_loss.item()

            # 更新判别器权重
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # 计算当前批次的PSNR
            batch_psnr = calculate_psnr(hr_images, generated_high_res)
            epoch_psnr += batch_psnr

            # 输出训练信息
            if (i + 1) % 10 == 0:
                print(f"Training epoch {epoch + 1}/{num_epochs}, num_steps {i + 1}/{len(train_loader)},"
                      f" 生成器损失: {generator_loss.item():.4f}, 判别器损失:{discriminator_loss.item():.4f}，"
                      f" 真实图像损失:{real_loss.item():.4f}, 生成图像的损失:{fake_loss.item():.4f},"
                      f" PSNR: {batch_psnr:.4f} dB")  # 打印损失信息和PSNR

        epoch_psnr /= len(train_loader)
        psnr_values.append(epoch_psnr)

        # 记录损失
        losses['生成器损失'].append(generator_loss_total / len(train_loader))
        losses['判别器损失'].append(discriminator_loss_total / len(train_loader))
        losses['真实图像损失'].append(real_loss_total / len(train_loader))
        losses['生成图像损失'].append(fake_loss_total / len(train_loader))
        losses['内容损失'].append(content_loss_value_total / len(train_loader))
        losses['对抗损失'].append(adversarial_loss_value_total / len(train_loader))

        # 保存模型和生成的图像
        with torch.no_grad():
            generator.eval()
            hr_images = next(iter(train_loader))[1][:8].to(device)
            lr_images = nn.functional.interpolate(hr_images, scale_factor=1 / scale_factor, mode='bicubic')
            fake_hr_images = generator(lr_images)
            save_image(hr_images, os.path.join(save_path, f'hr_images_{epoch + 1}.png'))
            save_image(lr_images, os.path.join(save_path, f'lr_images_{epoch + 1}.png'))
            save_image(fake_hr_images, os.path.join(save_path, f'fake_hr_images_{epoch + 1}.png'))
            print("保存生成的图像成功！")
            if (epoch + 1) % 5 == 0:
                torch.save(generator.state_dict(), os.path.join(save_path, f'generator_{epoch + 1}.pth'))
                torch.save(discriminator.state_dict(), os.path.join(save_path, f'discriminator_{epoch + 1}.pth'))
                print(f"保存模型成功，epoch: {epoch + 1}")

    print("Training complete!")
    save_dir = r'E:\Develop\Super_Resolution\Super_Resolution\result\SRGAN\96_5e-5_ep_200\loss'
    plot_losses(losses, save_dir)
    plot_individual_losses(losses, save_dir)
    plot_psnr(psnr_values, save_dir)  # 绘制PSNR曲线

if __name__ == '__main__':
    batch_size = 1
    lr = 5e-5
    num_epochs = 200
    scale_factor = 4
    root_dir = r'E:\Develop\Super_Resolution\Super_Resolution\data\DIV2K_train_HR'
    train(batch_size, lr, num_epochs, scale_factor, root_dir)