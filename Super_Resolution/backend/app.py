from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from models import Generator
import numpy as np
import math
import uuid
import io
from main import create_comparison_image
import base64

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# 配置上传文件夹和结果文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_model(device):
    """加载预训练的SRGAN模型"""
    model = Generator()
    model_dict = torch.load(r'E:\Develop\Super_Resolution\Super_Resolution\result\SRGAN\96_5e-5\generator_100.pth',
                            map_location=device)
    model.load_state_dict(model_dict)
    model.eval()
    model = model.to(device)
    print(f"生成器模型加载完毕，使用设备: {device}")
    return model


model = load_model(device)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel) - 10 * math.log10(mse)
    return psnr


def calculate_ssim(img1, img2, k1=0.01, k2=0.03, L=1.0):
    """计算SSIM"""
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    return ssim


def process_image(input_image, model, device):
    """处理单张图片"""
    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 对输入图像进行预处理
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # 使用模型进行超分辨率重建
    with torch.no_grad():
        sr_tensor = model(input_tensor)

    # 将tensor转换为PIL图像
    sr_img = sr_tensor.squeeze(0).cpu().detach()
    sr_img = (sr_img + 1) / 2.0  # 反归一化到[0,1]范围
    sr_img = torch.clamp(sr_img, 0, 1)  # 确保值在[0,1]范围内
    sr_img = (sr_img * 255.0).to(torch.uint8)  # 转换到[0,255]范围
    sr_img = sr_img.numpy().transpose(1, 2, 0)
    sr_img = Image.fromarray(sr_img)

    return sr_img


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 读取上传的图片
        image_bytes = file.read()
        input_image = Image.open(io.BytesIO(image_bytes))

        # 处理图片
        sr_img = process_image(input_image, model, device)

        # 创建对比图
        comparison_img = create_comparison_image(input_image, input_image, sr_img, 0, 0)

        # 将对比图转换为字节流
        img_byte_arr = io.BytesIO()
        comparison_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return send_file(
            img_byte_arr,
            mimetype='image/png',
            as_attachment=True,
            download_name='comparison_result.png'
        )

    except Exception as e:
        print(f"处理文件 {file.filename} 时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/results', methods=['GET'])
def get_results():
    results = []
    for filename in os.listdir(app.config['RESULT_FOLDER']):
        if filename.startswith('sr_'):
            original_filename = filename[3:]  # 移除'sr_'前缀
            results.append({
                'originalUrl': f'/uploads/{original_filename}',
                'srUrl': f'/results/{filename}',
                'psnr': 0,  # 这些值需要从某个地方获取或重新计算
                'ssim': 0
            })
    return jsonify(results)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/api/process-image', methods=['POST'])
def process_image_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # 保证RGB
        sr_img = process_image(input_image, model, device)
        sr_img = sr_img.convert('RGB')  # 保证RGB

        input_buffer = io.BytesIO()
        input_image.save(input_buffer, format='PNG')
        input_base64 = base64.b64encode(input_buffer.getvalue()).decode('utf-8')

        sr_buffer = io.BytesIO()
        sr_img.save(sr_buffer, format='PNG')
        sr_base64 = base64.b64encode(sr_buffer.getvalue()).decode('utf-8')

        # 调试输出
        print('原图base64前100:', input_base64[:100])
        print('超分base64前100:', sr_base64[:100])

        return jsonify({
            'original': input_base64,
            'super_resolution': sr_base64
        })

    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download', methods=['POST'])
def download_image():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # 将base64图片数据转换回二进制
        image_data = base64.b64decode(data['image'])
        image_buffer = io.BytesIO(image_data)

        return send_file(
            image_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='super_resolution_result.png'
        )

    except Exception as e:
        print(f"下载图片时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

