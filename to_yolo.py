import cv2
import os
import glob
import numpy as np
from pathlib import Path

def mask_to_yolo_seg(mask_path, class_id=0):
    """
    将二值掩码PNG图像转换为YOLO分割格式的多边形坐标列表。
    
    参数:
        mask_path: 掩码图片的路径。
        class_id: 目标的类别ID（默认为0）。
    
    返回:
        segments: 列表，每个元素是一个列表，代表一个多边形实例的归一化坐标 [class_id, x1, y1, x2, y2, ...]。
                  如果掩码为空或没有找到轮廓，则返回空列表。
    """
    # 读取掩码图片 (灰度图)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告: 无法读取掩码文件 {mask_path}")
        return []
    
    # 将掩码二值化 (假设非零像素为前景)
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segments = []
    height, width = mask.shape[:2]
    
    for contour in contours:
        # 轮廓必须至少有三个点才能构成多边形
        if contour.size < 6:  # 每个点有x,y，所以size=点数量*2
            continue
        
        # 压缩轮廓，去除冗余点，简化多边形 (epsilon 参数可调整)
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 将多边形点展平为一维数组，并归一化
        if len(approx) >= 3:  # 确保简化后仍然是多边形
            polygon = approx.flatten().astype(float)
            polygon[0::2] /= width   # 归一化 x 坐标
            polygon[1::2] /= height  # 归一化 y 坐标
            
            # 检查坐标是否在0-1范围内
            if (polygon >= 0).all() and (polygon <= 1).all():
                segment = [class_id] + polygon.tolist()
                segments.append(segment)
            else:
                print(f"警告: 在 {mask_path} 中发现归一化坐标超出[0,1]范围，已跳过。")
    
    return segments

def process_dataset(base_dir, subset='train'):
    """
    处理一个子集（如train, val, test）。
    
    参数:
        base_dir: 数据集根目录 (data_wound_seg)。
        subset: 子集名称 ('train', 'val', 或 'test')。
    """
    images_dir = os.path.join(base_dir, f'{subset}_images')
    masks_dir = os.path.join(base_dir, f'{subset}_masks')
    
    # 检查目录是否存在
    if not os.path.isdir(images_dir):
        print(f"错误: 图片目录不存在 {images_dir}")
        return
    if not os.path.isdir(masks_dir):
        print(f"错误: 掩码目录不存在 {masks_dir}")
        return
    
    # 获取所有掩码文件
    mask_files = glob.glob(os.path.join(masks_dir, '*.png'))
    print(f"正在处理子集 '{subset}'，找到 {len(mask_files)} 个掩码文件。")
    
    for mask_path in mask_files:
        # 生成对应的图片文件名（假设图片和掩码文件名相同）
        mask_name = os.path.basename(mask_path)
        # 对应的图片可能也是png，这里按同名处理。如果图片是其他格式（如jpg），请修改后缀。
        image_name = mask_name  # 假设图片文件名与掩码完全相同
        image_path = os.path.join(images_dir, image_name)
        
        # 可选：检查对应的图片文件是否存在
        if not os.path.isfile(image_path):
            # 如果不存在，尝试查找不带后缀或不同后缀的文件
            name_without_ext = os.path.splitext(mask_name)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                possible_image_path = os.path.join(images_dir, name_without_ext + ext)
                if os.path.isfile(possible_image_path):
                    image_path = possible_image_path
                    break
            else:
                print(f"警告: 未找到掩码 {mask_name} 对应的图片文件，跳过。")
                continue
        
        # 转换掩码为YOLO分割标注
        segments = mask_to_yolo_seg(mask_path, class_id=0)  # 假设您的类别ID是0
        
        # 准备输出标注文件路径 (与图片同名，但放在 masks_dir 同级，或您指定的位置)
        # 这里选择在 masks_dir 同级创建一个 'labels' 文件夹来存放 .txt 文件，以符合YOLO常见结构
        labels_dir = os.path.join(base_dir, f'{subset}_labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        label_name = os.path.splitext(mask_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        # 将标注写入文件
        with open(label_path, 'w') as f:
            for segment in segments:
                # 将列表转换为字符串，用空格分隔
                line = ' '.join(map(str, segment))
                f.write(line + '\n')
        
        # 可选：输出进度
        if len(mask_files) > 10 and (mask_files.index(mask_path) + 1) % 50 == 0:
            print(f"  已处理 {mask_files.index(mask_path) + 1}/{len(mask_files)}")

def main():
    # 设置您的数据根目录路径
    base_data_dir = './data_seg'  # 请修改为您的实际路径
    
    # 处理训练集、验证集、测试集
    for subset in ['train', 'val', 'test']:
        print(f"\n=== 开始处理 {subset} 集 ===")
        process_dataset(base_data_dir, subset)
        print(f"=== {subset} 集处理完成 ===\n")
    
    print("所有转换完成！")
    print("生成的标注文件位于：")
    print(f"  {os.path.join(base_data_dir, 'train_labels')}")
    print(f"  {os.path.join(base_data_dir, 'val_labels')}")
    print(f"  {os.path.join(base_data_dir, 'test_labels')}")

if __name__ == '__main__':
    main()