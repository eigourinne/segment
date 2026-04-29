import cv2
import os
import glob
import numpy as np
import pandas as pd
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def mask_to_yolo_seg(mask_path, class_id=0):
    """
    将二值掩码PNG图像转换为YOLO分割格式的多边形坐标列表。
    此函数直接来源于您提供的 `to_yolo.py` 文档。
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告: 无法读取掩码文件 {mask_path}")
        return []
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    height, width = mask.shape[:2]
    for contour in contours:
        if contour.size < 6:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            polygon = approx.flatten().astype(float)
            polygon[0::2] /= width
            polygon[1::2] /= height
            if (polygon >= 0).all() and (polygon <= 1).all():
                segment = [class_id] + polygon.tolist()
                segments.append(segment)
            else:
                print(f"警告: 在 {mask_path} 中发现归一化坐标超出[0,1]范围，已跳过。")
    return segments

def load_and_map_filenames(base_dir, table_path):
    """
    加载映射表，并建立 原始文件名 <-> 新文件名 的双向字典。
    """
    table_full_path = os.path.join(base_dir, table_path)
    try:
        # 读取Excel，假设前两列为原始名和新名
        df = pd.read_excel(table_full_path, header=None)
        # 简单处理：将前两列分别视为原始名和新名
        origin_to_new = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        new_to_origin = {v: k for k, v in origin_to_new.items()}
        print(f"成功从 {table_path} 加载了 {len(origin_to_new)} 个文件名映射。")
        return origin_to_new, new_to_origin
    except Exception as e:
        print(f"错误: 无法读取或解析映射表 {table_path}。错误信息: {e}")
        return {}, {}

def collect_dataset(base_dir, subset):
    """
    收集指定子集（train 或 test）的所有图片和掩码对。
    返回一个列表，每个元素是 (图片路径, 掩码路径) 的元组。
    """
    images_dir = os.path.join(base_dir, f'{subset}_images')
    masks_dir = os.path.join(base_dir, f'{subset}_masks')
    
    if not os.path.isdir(images_dir):
        print(f"错误: 图片目录不存在 {images_dir}")
        return []
    if not os.path.isdir(masks_dir):
        print(f"错误: 掩码目录不存在 {masks_dir}")
        return []
    
    all_pairs = []
    
    # 获取所有图片文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    print(f"在 {images_dir} 中找到 {len(image_files)} 个图片文件")
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # 在掩码目录中查找对应的掩码文件
        # 先尝试使用相同的文件名（相同扩展名）
        mask_path = os.path.join(masks_dir, img_name_no_ext + '.png')
        
        if not os.path.isfile(mask_path):
            # 如果找不到，尝试使用图片文件名（不带扩展名）加上.png
            mask_path = os.path.join(masks_dir, img_name_no_ext + '.png')
            
        if not os.path.isfile(mask_path):
            # 如果还找不到，尝试查找任何扩展名的同名文件
            mask_candidates = glob.glob(os.path.join(masks_dir, img_name_no_ext + '.*'))
            if mask_candidates:
                mask_path = mask_candidates[0]
            else:
                print(f"警告: 未找到图片 {img_name} 对应的掩码文件，跳过。")
                continue
        
        all_pairs.append((img_path, mask_path))
    
    print(f"子集 {subset} 收集到 {len(all_pairs)} 个有效样本。")
    return all_pairs

def split_and_convert(base_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    主流程函数：
    1. 合并收集 train 和 test 的所有数据。
    2. 按 8:1:1 随机划分为 train, val, test。
    3. 为每个划分创建YOLO目录，转换并保存图片和标签。
    """
    # 1. 合并数据
    all_data_pairs = []
    for subset in ['train', 'test']:
        pairs = collect_dataset(base_dir, subset)
        all_data_pairs.extend(pairs)
    
    if not all_data_pairs:
        print("错误: 未收集到任何有效的数据对。请检查目录结构和文件。")
        return
    
    print(f"总共收集到 {len(all_data_pairs)} 个样本。")
    
    # 2. 随机划分 (8:1:1)
    # 先拆出测试集（占总体的10%）
    train_val_pairs, test_pairs = train_test_split(
        all_data_pairs, test_size=val_ratio, random_state=seed
    )
    # 从剩下的90%中，拆出验证集（占 train_val_pairs 的 1/9，即总体的10%）
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, test_size=val_ratio/(train_ratio+val_ratio), random_state=seed
    )
    
    print(f"划分结果: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    # 3. 为每个子集创建目录并处理
    subset_dict = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}
    for subset_name, pairs in subset_dict.items():
        print(f"\n=== 正在处理 {subset_name} 集 ===")
        # 创建YOLO格式的输出目录
        subset_output_dir = os.path.join(output_dir, subset_name)
        images_output_dir = os.path.join(subset_output_dir, 'images')
        labels_output_dir = os.path.join(subset_output_dir, 'labels')
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)
        
        processed_count = 0
        for img_path, mask_path in pairs:
            # 生成唯一的新文件名（使用递增序号）
            new_filename_base = f"{subset_name}_{processed_count:06d}"
            img_ext = os.path.splitext(img_path)[1]
            new_img_filename = new_filename_base + img_ext
            new_label_filename = new_filename_base + '.txt'
            
            new_img_path = os.path.join(images_output_dir, new_img_filename)
            new_label_path = os.path.join(labels_output_dir, new_label_filename)
            
            # 复制图片文件
            try:
                shutil.copy2(img_path, new_img_path)
            except Exception as e:
                print(f"警告: 复制图片 {img_path} 到 {new_img_path} 失败: {e}")
                continue
            
            # 转换掩码并保存标签
            segments = mask_to_yolo_seg(mask_path, class_id=0)
            if segments:
                with open(new_label_path, 'w') as f:
                    for segment in segments:
                        line = ' '.join(map(str, segment))
                        f.write(line + '\n')
            else:
                # 如果没有分割目标，创建一个空的标签文件
                open(new_label_path, 'w').close()
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  已处理 {processed_count}/{len(pairs)} 个样本")
        
        print(f"=== {subset_name} 集处理完成，共 {processed_count} 个样本 ===")
    
    print(f"\n所有转换完成！")
    print(f"YOLO格式数据集已保存至: {output_dir}")
    print("每个子集目录结构示例：")
    print(f"  {output_dir}/train/images/")
    print(f"  {output_dir}/train/labels/")

if __name__ == '__main__':
    # 配置路径
    base_data_dir = './data_seg'  # 根据您的实际路径修改
    output_root_dir = './yolo_dataset'  # 最终输出目录
    
    # 设置随机种子以确保可重复性
    random_seed = 42
    
    # 执行主流程
    split_and_convert(base_data_dir, output_root_dir, train_ratio=0.8, val_ratio=0.1, seed=random_seed)