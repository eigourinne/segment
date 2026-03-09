"""
数据处理模块 - 处理数据集加载、验证和预处理
"""
import yaml
import json
import os
import shutil
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
from config import config

logger = logging.getLogger(__name__)

class YOLODataset:
    """YOLO格式数据集管理类"""
    
    def __init__(self, data_yaml_path: str = None):
        self.data_yaml_path = Path(data_yaml_path) if data_yaml_path else config.DATA_YAML
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """加载data.yaml配置文件"""
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"找不到配置文件: {self.data_yaml_path}")
        
        with open(self.data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 确保使用绝对路径
        config['path'] = str(Path(config['path']).absolute())
        return config
    
    def verify(self) -> bool:
        """
        验证数据集完整性
        
        Returns:
            bool: 数据集是否有效
        """
        try:
            print(f"验证数据集: {self.data_yaml_path}")
            print("=" * 50)
            
            # 检查配置文件
            if not self.data_yaml_path.exists():
                logger.error(f"配置文件不存在: {self.data_yaml_path}")
                return False
            
            # 检查数据集根目录
            data_path = Path(self.config['path'])
            if not data_path.exists():
                logger.error(f"数据集根目录不存在: {data_path}")
                return False
            
            # 检查训练集和验证集
            for split in ['train', 'val']:
                if split not in self.config:
                    logger.warning(f"配置中缺少{split}路径")
                    continue
                    
                img_dir = data_path / self.config[split]
                label_dir = data_path / "labels" / split
                
                print(f"\n{split.upper()}集检查:")
                print(f"  图片目录: {img_dir}")
                print(f"  标签目录: {label_dir}")
                
                # 检查目录
                if not img_dir.exists():
                    logger.error(f"图片目录不存在: {img_dir}")
                    return False
                    
                if not label_dir.exists():
                    logger.error(f"标签目录不存在: {label_dir}")
                    return False
                
                # 检查文件
                img_files = list(img_dir.glob("*"))
                label_files = list(label_dir.glob("*.txt"))
                
                print(f"  图片文件: {len(img_files)} 个")
                print(f"  标签文件: {len(label_files)} 个")
                
                # 检查文件对应关系
                img_stems = {f.stem for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
                label_stems = {f.stem for f in label_files}
                
                missing_labels = img_stems - label_stems
                missing_images = label_stems - img_stems
                
                if missing_labels:
                    print(f"  ⚠️ 有 {len(missing_labels)} 张图片没有对应的标签")
                    for name in list(missing_labels)[:3]:
                        print(f"    - {name}")
                        
                if missing_images:
                    print(f"  ⚠️ 有 {len(missing_images)} 个标签没有对应的图片")
                    for name in list(missing_images)[:3]:
                        print(f"    - {name}")
            
            # 检查类别信息
            print(f"\n类别信息:")
            print(f"  类别数量: {self.config.get('nc', 0)}")
            print(f"  类别名称: {self.config.get('names', {})}")
            
            print("\n✅ 数据集验证通过!")
            return True
            
        except Exception as e:
            logger.error(f"数据集验证失败: {e}")
            return False
    
    def get_split_info(self, split: str = 'train') -> Dict:
        """获取指定数据集的统计信息"""
        data_path = Path(self.config['path'])
        img_dir = data_path / self.config[split]
        label_dir = data_path / "labels" / split
        
        img_files = [f for f in img_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        label_files = [f for f in label_dir.iterdir() 
                      if f.is_file() and f.suffix == '.txt']
        
        # 统计标签信息
        class_counts = {}
        bbox_info = {
            'total': 0,
            'avg_per_image': 0,
            'size_stats': []
        }
        
        for label_file in label_files[:100]:  # 只检查前100个文件加快速度
            with open(label_file, 'r') as f:
                lines = f.readlines()
                bbox_info['total'] += len(lines)
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        if img_files:
            bbox_info['avg_per_image'] = bbox_info['total'] / len(img_files)
        
        return {
            'split': split,
            'image_count': len(img_files),
            'label_count': len(label_files),
            'class_distribution': class_counts,
            'bbox_statistics': bbox_info
        }
    
    def create_dataset_yaml(self, output_path: str = None) -> Path:
        """创建data.yaml配置文件"""
        if output_path is None:
            output_path = config.DATA_DIR / "data.yaml"
        
        yaml_content = {
            'path': str(Path(self.config['path']).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.config.get('nc', 1),
            'names': self.config.get('names', {0: 'lung'})
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        print(f"✅ data.yaml 已创建: {output_path}")
        return Path(output_path)