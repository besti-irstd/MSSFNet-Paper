import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

# 导入你的模型
from model.MSSFNet import MSSFNet
from utils.checkpoint import load_checkpoint, extract_state_dict

PROJECT_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = PROJECT_DIR.parent / "MSSFNet-Code"


def parse_args():
    parser = ArgumentParser(description='Generate Prediction Images for MSSFNet')
    # 路径参数：请根据你的实际情况修改默认路径
    parser.add_argument('--img-dir', type=str, default=str(ORIGINAL_DIR / 'datasets' / 'IRSTD-1k' / 'images'),
                        help='Directory containing test images')
    parser.add_argument('--save-dir', type=str, default=str(PROJECT_DIR / 'outputs' / 'predictions'), help='Directory to save prediction images')
    parser.add_argument('--weight-path', type=str, default=str(ORIGINAL_DIR / 'weight' / 'IRSTD-1K' / 'weight.pkl'), help='Path to model weights')

    # 模型配置
    parser.add_argument('--deep-supervision', '--use-dhpf', dest='deep_supervision',
                        type=int, default=0, help='Enable auxiliary masks (0 or 1; default: 0); AWFE always runs; legacy alias: --use-dhpf')
    parser.add_argument('--gpu-id', type=str, default='0')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==> Loading Model...")
    # 初始化 MSSFNet，输入通道数为 3
    model = MSSFNet(3)

    # 安全读取并兼容旧模块命名、普通 state_dict 和训练 checkpoint。
    weight = load_checkpoint(args.weight_path, map_location='cpu')
    model.load_state_dict(extract_state_dict(weight))
    model.to(device)
    model.eval()  # 切换到测试模式

    # 创建保存预测图片的文件夹
    os.makedirs(args.save_dir, exist_ok=True)

    # 获取所有图片文件名
    img_names = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    print(f"==> Found {len(img_names)} images in {args.img_dir}")

    # 图像预处理：转换为 Tensor，归一化到 [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 控制辅助输出；AWFE 始终执行。
    tag = True if args.deep_supervision == 1 else False

    print("==> Start Generating Predictions...")
    with torch.no_grad():
        for img_name in tqdm(img_names):
            img_path = os.path.join(args.img_dir, img_name)
            save_path = os.path.join(args.save_dir, img_name)

            # 1. 读取图像 (统一转为RGB 3通道)
            img = Image.open(img_path).convert('RGB')
            original_size = img.size  # (width, height)

            # 2. 预处理并送入设备
            data = transform(img).unsqueeze(0).to(device)

            # 3. 模型推理
            # 注意：此处返回的 pred 是一个特征图 (B, 1, H, W)
            _, pred = model(data, tag)

            # 4. 后处理：生成预测图 (非二值掩码)
            # 使用 sigmoid 将输出映射到 [0, 1] 之间的概率值
            pred = torch.sigmoid(pred)

            # 压缩维度并转为 numpy 数组
            pred = pred.squeeze().cpu().numpy()

            # 将 [0, 1] 的概率图映射到 [0, 255] 的灰度图格式
            pred_img = (pred * 255).astype(np.uint8)

            # 如果需要，可以将预测图调整回原图尺寸 (防止网络内部改变了尺寸)
            if pred_img.shape[::-1] != original_size:
                pred_img = cv2.resize(pred_img, original_size, interpolation=cv2.INTER_LINEAR)

            # 5. 保存预测图
            cv2.imwrite(save_path, pred_img)

    print(f"==> All predictions saved to: {os.path.abspath(args.save_dir)}")


if __name__ == '__main__':
    main()