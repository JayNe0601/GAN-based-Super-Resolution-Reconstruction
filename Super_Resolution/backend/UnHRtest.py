import torch
import torchvision.transforms as transforms
from PIL import Image
from models import Generator
import os
import gc

def tensor_to_numpy(tensor):
    """将tensor转换为numpy数组"""
    return tensor.cpu().numpy().transpose(1, 2, 0)


def normalize_image(img):
    """将图像归一化到[0,1]范围"""
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)


def preprocess_image(image_path, max_size=512):
    """预处理图像，确保图像大小合适"""
    img = Image.open(image_path).convert('RGB')

    # 如果图像太大，进行缩放
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)

    return img

def get_image_name(image_path):
    """从路径中获取图片名称（不含扩展名）"""
    return os.path.splitext(os.path.basename(image_path))[0]

def test(input_image_path, save_dir, generator_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    image_name = get_image_name(input_image_path)
    print(f"处理图片: {image_name}")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

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

    try:
        # 加载并预处理输入图像
        input_img = preprocess_image(input_image_path)
        print(f"输入图像大小: {input_img.size}")

        # 对输入图像进行预处理
        input_tensor = transform(input_img).unsqueeze(0).to(device)

        # 将输入图像输入SRGAN模型进行超分辨率重建
        with torch.no_grad():
            sr_tensor = model(input_tensor)

        # 保存生成的高分辨率图像
        sr_img = sr_tensor.squeeze(0).cpu().detach().numpy()
        sr_img = (sr_img + 1) / 2.0 * 255.0
        sr_img = sr_img.clip(0, 255).astype('uint8')
        sr_img = Image.fromarray(sr_img.transpose(1, 2, 0))
        sr_save_path = os.path.join(save_dir, f'{image_name}_sr.png')
        sr_img.save(sr_save_path)

        print(f"处理完成！结果已保存到: {save_dir}")
        print(f"超分辨率图像: {sr_save_path}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU显存不足，请尝试使用更小的图像或增加GPU显存")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"发生错误: {str(e)}")
    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    # 输入图像路径
    input_image_path = r"E:\Develop\Super_Resolution\Super_Resolution\result\RUN\test\input3.png"
    # 保存目录
    save_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\RUN\test"
    # 模型路径
    generator_dir = r"E:\Develop\Super_Resolution\Super_Resolution\result\SRGAN\96_5e-5\generator_100.pth"

    test(input_image_path, save_dir, generator_dir)
