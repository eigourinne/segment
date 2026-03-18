import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import random
import yaml

# ==================== 配置部分 ====================
IMAGES_DIR = "images"  # 原图片目录
MASKS_DIR = "masks"    # 原掩码目录 (应为二值图，背景0，前景255或>0)
METADATA_PATH = "metadata.csv"  # 元数据文件路径

# 在metadata.csv中，用于匹配图片文件和标识类别的列名
IMAGE_FILE_COLUMN = "image"  # 图片文件名的列名
CATEGORY_COLUMNS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]  # 类别名称的列名

# 输出YOLO格式数据的主目录
YOLO_OUTPUT_DIR = "data-seg-yolo"
# 子目录结构
IMAGES_OUTPUT = os.path.join(YOLO_OUTPUT_DIR, "images")
LABELS_OUTPUT = os.path.join(YOLO_OUTPUT_DIR, "labels")

# 数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42  # 随机种子，确保划分可复现
# ==================== 配置结束 ====================

def create_directory_structure():
    """创建YOLO格式所需的目录结构"""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(IMAGES_OUTPUT, split), exist_ok=True)
        os.makedirs(os.path.join(LABELS_OUTPUT, split), exist_ok=True)
    print(f"目录结构已在 '{YOLO_OUTPUT_DIR}' 下创建。")

def mask_to_polygon(mask_img, img_width, img_height, epsilon_factor=0.005):
    """
    核心函数：从一张二值掩码图像计算YOLO实例分割格式的多边形标注。
    处理一张图中可能存在的多个实例（连通区域）。
    
    Args:
        mask_img: 二值掩码图像 (numpy array, 0为背景，非0为前景)
        img_width: 原图宽度
        img_height: 原图高度
        epsilon_factor: 多边形轮廓近似系数，用于减少点数。值越大，多边形越简化。
    
    Returns:
        polygons_list: 列表，每个元素是一个列表，代表一个实例的多边形坐标 [x1, y1, x2, y2, ...]
                      坐标是归一化的。
    """
    polygons_list = []
    
    # 1. 确保掩码是二值图
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)
    
    # 2. 找到掩码中所有轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # 跳过太小的轮廓（可能是噪声）
        if cv2.contourArea(cnt) < 10:
            continue
            
        # 3. 轮廓近似，以减少点的数量（重要，防止点数过多）
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 4. 检查近似后的多边形是否至少是三角形 (3个点)
        if len(approx) < 3:
            # 如果点数太少，用原轮廓或跳过
            # 这里选择用最小外接矩形生成一个近似的四边形，保证至少3个点
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            approx = np.int0(box).reshape(-1, 1, 2)
            if len(approx) < 3:
                continue  # 如果仍然不够，跳过此轮廓
        
        # 5. 将轮廓点展平并归一化
        # approx 形状: (n, 1, 2) -> 转换为 (n, 2)
        points = approx.reshape(-1, 2).astype(float)
        points[:, 0] /= img_width   # 归一化 x
        points[:, 1] /= img_height  # 归一化 y
        
        # 6. 确保坐标在[0,1]范围内，并限制精度
        points = np.clip(points, 0.0, 1.0)
        points = np.round(points, 6)
        
        # 7. 展平为 [x1, y1, x2, y2, ...] 的一维列表
        polygon_flat = points.flatten().tolist()
        polygons_list.append(polygon_flat)
    
    return polygons_list

def process_dataset():
    create_directory_structure()
    
    # 加载元数据
    df = pd.read_csv(METADATA_PATH)
    assert IMAGE_FILE_COLUMN in df.columns, f"CSV中未找到图片ID列 '{IMAGE_FILE_COLUMN}'"
    
    # 建立类别名到ID的映射
    class_to_id = {cls: idx for idx, cls in enumerate(CATEGORY_COLUMNS)}
    unique_classes = CATEGORY_COLUMNS
    print(f"处理 {len(unique_classes)} 个类别: {class_to_id}")
    
    # 保存类别映射文件
    with open(os.path.join(YOLO_OUTPUT_DIR, "classes.txt"), "w") as f:
        for cls in unique_classes:
            f.write(cls + "\n")
    
    data_records = []  # 用于存储成功处理的数据
    failed_files = []  # 记录失败的文件
    
    print("开始处理数据集...")
    for idx, row in df.iterrows():
        img_id = row[IMAGE_FILE_COLUMN]  # 例如: ISIC_0024306
        
        # 1. 确定该图片的类别
        # 从多标签列中提取所有激活的类别ID
        activated_class_ids = []
        for cls_name in CATEGORY_COLUMNS:
            if row[cls_name] == 1.0:
                activated_class_ids.append(class_to_id[cls_name])
        
        # 如果没有激活的类别，跳过
        if not activated_class_ids:
            print(f"警告: {img_id} 没有激活的类别标签，跳过。")
            failed_files.append((img_id, "无激活类别"))
            continue
        
        # **关键决策**：一个二值掩码对应多个类别标签，如何分配？
        # 方案：选择第一个激活的类别作为该图片中所有实例的类别。
        # 这是因为实例分割要求每个实例有唯一类别，而掩码无法区分不同类别的区域。
        # 如果您有更精确的每实例类别信息，需要修改此处逻辑。
        assigned_class_id = activated_class_ids[0]
        if len(activated_class_ids) > 1:
            print(f"注意: 图片 {img_id} 有多个激活类别 {activated_class_ids}。将使用第一个类别 {assigned_class_id} ({CATEGORY_COLUMNS[assigned_class_id]}) 作为所有实例的标签。")
        
        # 2. 构建文件路径
        img_file = img_id + ".jpg"  # 根据您的实际文件扩展名调整
        mask_file = img_id + ".png"  # 根据您的实际掩码文件扩展名调整
        
        img_path = os.path.join(IMAGES_DIR, img_file)
        mask_path = os.path.join(MASKS_DIR, mask_file)
        
        # 3. 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"错误: 图片文件不存在 {img_path}")
            failed_files.append((img_id, f"图片缺失: {img_file}"))
            continue
        if not os.path.exists(mask_path):
            print(f"错误: 掩码文件不存在 {mask_path}")
            failed_files.append((img_id, f"掩码缺失: {mask_file}"))
            continue
        
        # 4. 读取图片和掩码
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 无法读取图片 {img_path}")
            failed_files.append((img_id, f"图片读取失败: {img_file}"))
            continue
        img_height, img_width = img.shape[:2]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"错误: 无法读取掩码 {mask_path}")
            failed_files.append((img_id, f"掩码读取失败: {mask_file}"))
            continue
        
        # 5. 从掩码中提取多边形
        polygons = mask_to_polygon(mask, img_width, img_height, epsilon_factor=0.005)
        
        # 6. 检查是否提取到有效的多边形
        if not polygons:
            print(f"警告: 在 {img_id} 的掩码中未找到有效轮廓，跳过。")
            failed_files.append((img_id, "掩码中无有效轮廓"))
            continue
        
        # 7. 为每个多边形实例构建YOLO格式的标注行
        yolo_annotation_lines = []
        for polygon in polygons:
            # 格式: class_id x1 y1 x2 y2 ...
            polygon_str = ' '.join([f'{coord:.6f}' for coord in polygon])
            annotation_line = f"{assigned_class_id} {polygon_str}"
            yolo_annotation_lines.append(annotation_line)
        
        base_name = img_id
        data_records.append({
            "base_name": base_name,
            "yolo_annotations": yolo_annotation_lines,  # 列表，每个元素是一个实例的标注行
            "img_path": img_path,
            "img_width": img_width,
            "img_height": img_height
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1} 个样本...")
    
    print(f"\n处理完成。成功: {len(data_records)} 个样本，失败: {len(failed_files)} 个。")
    if failed_files:
        failed_log_path = os.path.join(YOLO_OUTPUT_DIR, "failed_files.log")
        with open(failed_log_path, "w") as f:
            f.write("image_id,reason\n")
            for img_id, reason in failed_files:
                f.write(f"{img_id},{reason}\n")
        print(f"失败详情已保存至 '{failed_log_path}'")
    
    # 8. 划分数据集
    if len(data_records) == 0:
        print("错误: 没有成功处理任何数据，请检查配置和文件。")
        return
    
    # 先分割出训练+验证集 和 测试集
    train_val_data, test_data = train_test_split(
        data_records, test_size=TEST_RATIO, random_state=RANDOM_SEED, shuffle=True
    )
    # 再从训练+验证集中分割出训练集和验证集
    val_relative_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_relative_ratio, random_state=RANDOM_SEED, shuffle=True
    )
    
    print(f"数据集划分完成 -> 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
    
    # 9. 复制文件并生成标签文件
    def write_split_data(split_name, data_list):
        for record in data_list:
            base_name = record["base_name"]
            src_img_path = record["img_path"]
            yolo_lines = record["yolo_annotations"]
            
            # 目标路径
            dst_img_dir = os.path.join(IMAGES_OUTPUT, split_name)
            dst_label_dir = os.path.join(LABELS_OUTPUT, split_name)
            
            # 保留原图片扩展名
            _, src_ext = os.path.splitext(src_img_path)
            dst_img_path = os.path.join(dst_img_dir, base_name + src_ext)
            dst_label_path = os.path.join(dst_label_dir, base_name + ".txt")
            
            # 复制图片
            shutil.copy2(src_img_path, dst_img_path)
            # 写入标签文件 (每个实例一行)
            with open(dst_label_path, "w") as f:
                for line in yolo_lines:
                    f.write(line + "\n")
    
    write_split_data("train", train_data)
    write_split_data("val", val_data)
    write_split_data("test", test_data)
    
    # 10. 创建YOLO数据集配置文件 (data.yaml)
    yaml_data = {
        "path": os.path.abspath(YOLO_OUTPUT_DIR),  # 数据集根目录的绝对路径
        "train": "images/train",  # 相对path的路径
        "val": "images/val",
        "test": "images/test",
        "nc": len(unique_classes),  # 类别数
        "names": unique_classes  # 类别名列表
    }
    
    yaml_path = os.path.join(YOLO_OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n转换完成！")
    print(f"YOLO格式数据集已保存至: {os.path.abspath(YOLO_OUTPUT_DIR)}")
    print(f"数据集结构:")
    print(f"  {YOLO_OUTPUT_DIR}/")
    print(f"  ├── images/train/")
    print(f"  ├── images/val/")
    print(f"  ├── images/test/")
    print(f"  ├── labels/train/")
    print(f"  ├── labels/val/")
    print(f"  ├── labels/test/")
    print(f"  ├── classes.txt")
    print(f"  ├── data.yaml")
    print(f"  └── failed_files.log (如果存在)")
    print(f"\n您可以在训练中使用以下配置:")
    print(f"  model.train(data='{yaml_path}', ...)")

if __name__ == "__main__":
    process_dataset()