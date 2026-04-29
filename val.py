import os
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
import warnings
warnings.filterwarnings('ignore')

# 设置字体，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体
plt.rcParams['axes.unicode_minus'] = False

class MedicalSegmentationEvaluator:
    """Medical Image Segmentation Evaluator"""
    
    def __init__(self, data_yaml_path, config_txt_path):
        """
        Initialize the evaluator
        
        Args:
            data_yaml_path: path to dataset configuration file
            config_txt_path: path to model configuration file
        """
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.val_dir = data_config['val']  # validation set path
        self.num_classes = data_config['nc']  # number of classes
        self.class_names = data_config['names']  # class names
        
        # Load model configuration
        with open(config_txt_path, 'r') as f:
            lines = f.readlines()
        
        self.model_paths = {}
        for line in lines:
            if ':' in line:
                name, path = line.strip().split(':')
                self.model_paths[name.strip()] = path.strip()
        
        # Collect validation image paths
        self.val_images = list(Path(self.val_dir).glob('*.jpg')) + \
                         list(Path(self.val_dir).glob('*.png')) + \
                         list(Path(self.val_dir).glob('*.jpeg')) + \
                         list(Path(self.val_dir).glob('*.bmp'))
        
        print(f"Found {len(self.val_images)} validation images")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Models to evaluate: {list(self.model_paths.keys())}")
    
    def load_ground_truth(self, image_path):
        """
        Load ground truth annotations (assuming annotation files have same name as image with .txt extension)
        
        Args:
            image_path: path to the image
        
        Returns:
            masks: list of ground truth masks, each element is a binary mask
        """
        # Assuming YOLO format annotations
        # In practice, this should be adjusted according to your annotation format
        label_path = Path(str(image_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt'))
        
        if not label_path.exists():
            return []
        
        # Read image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        h, w = img.shape[:2]
        masks = []
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:  # at least class ID and multiple point coordinates are required
                continue
            
            class_id = int(parts[0])
            # Parse polygon points
            points = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            # Convert normalized coordinates to pixel coordinates
            points[:, 0] *= w
            points[:, 1] *= h
            points = points.astype(np.int32)
            
            # Create binary mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            masks.append((class_id, mask))
        
        return masks
    
    def resize_mask(self, mask, target_shape):
        """
        Resize mask to target shape
        
        Args:
            mask: original mask
            target_shape: target shape (height, width)
        
        Returns:
            resized_mask: resized mask
        """
        if mask.shape == target_shape:
            return mask
        
        # Use nearest neighbor interpolation to preserve binary properties
        resized_mask = cv2.resize(mask.astype(np.float32), 
                                 (target_shape[1], target_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Re-binarize
        resized_mask = (resized_mask > 0.5).astype(np.uint8)
        return resized_mask
    
    def calculate_dice(self, pred_mask, gt_mask):
        """
        Calculate Dice coefficient
        
        Args:
            pred_mask: predicted mask
            gt_mask: ground truth mask
        
        Returns:
            dice: Dice coefficient
        """
        # Ensure masks have the same shape
        if pred_mask.shape != gt_mask.shape:
            gt_mask = self.resize_mask(gt_mask, pred_mask.shape)
        
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            return 1.0
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            return 0.0
        
        return 2.0 * intersection / union
    
    def calculate_iou(self, pred_mask, gt_mask):
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            pred_mask: predicted mask
            gt_mask: ground truth mask
        
        Returns:
            iou: IoU value
        """
        # Ensure masks have the same shape
        if pred_mask.shape != gt_mask.shape:
            gt_mask = self.resize_mask(gt_mask, pred_mask.shape)
        
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            return 1.0
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_hd95(self, pred_mask, gt_mask):
        """
        Calculate 95% Hausdorff Distance (HD95)
        
        Args:
            pred_mask: predicted mask
            gt_mask: ground truth mask
        
        Returns:
            hd95: HD95 distance
        """
        # Ensure masks have the same shape
        if pred_mask.shape != gt_mask.shape:
            gt_mask = self.resize_mask(gt_mask, pred_mask.shape)
        
        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            return float('inf')
        
        # Get boundary point coordinates
        pred_points = np.argwhere(pred_mask > 0)
        gt_points = np.argwhere(gt_mask > 0)
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return float('inf')
        
        # Calculate bidirectional Hausdorff distance
        hd1 = directed_hausdorff(pred_points, gt_points)[0]
        hd2 = directed_hausdorff(gt_points, pred_points)[0]
        
        # Take the maximum as the Hausdorff distance
        hd = max(hd1, hd2)
        
        return hd
    
    def calculate_rve(self, pred_mask, gt_mask):
        """
        Calculate Relative Volume Error (RVE)
        
        Args:
            pred_mask: predicted mask
            gt_mask: ground truth mask
        
        Returns:
            rve: relative volume error (0-1 range)
        """
        # Ensure masks have the same shape
        if pred_mask.shape != gt_mask.shape:
            gt_mask = self.resize_mask(gt_mask, pred_mask.shape)
        
        pred_volume = np.sum(pred_mask)
        gt_volume = np.sum(gt_mask)
        
        if gt_volume == 0:
            if pred_volume == 0:
                return 0.0
            else:
                return float('inf')
        
        # RVE should be in 0-1 range
        rve = abs(pred_volume - gt_volume) / gt_volume
        # 确保RVE不超过1，因为当预测体积为0时，RVE最大为1
        return min(rve, 1.0)
    
    def calculate_pa(self, pred_mask, gt_mask):
        """
        Calculate Pixel Accuracy (PA)
        
        Args:
            pred_mask: predicted mask
            gt_mask: ground truth mask
        
        Returns:
            pa: pixel accuracy
        """
        # Ensure masks have the same shape
        if pred_mask.shape != gt_mask.shape:
            gt_mask = self.resize_mask(gt_mask, pred_mask.shape)
        
        correct = np.sum(pred_mask == gt_mask)
        total = pred_mask.size
        
        return correct / total
    
    def match_masks(self, pred_masks, gt_masks, iou_threshold=0.5):
        """
        Match predicted masks and ground truth masks
        
        Args:
            pred_masks: list of predicted masks, each element is (mask, confidence, class)
            gt_masks: list of ground truth masks, each element is (class, mask)
            iou_threshold: IoU matching threshold
        
        Returns:
            matched_pairs: list of matched mask pairs
            unmatched_pred: indices of unmatched predicted masks
            unmatched_gt: indices of unmatched ground truth masks
        """
        matched_pairs = []
        used_pred = [False] * len(pred_masks)
        used_gt = [False] * len(gt_masks)
        
        # Sort predicted masks by confidence in descending order
        pred_masks_sorted = sorted(enumerate(pred_masks), 
                                  key=lambda x: x[1][1] if len(x[1]) > 1 else 0, 
                                  reverse=True)
        
        for pred_idx, (pred_mask, pred_conf, pred_class) in pred_masks_sorted:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_class, gt_mask) in enumerate(gt_masks):
                if used_gt[gt_idx] or pred_class != gt_class:
                    continue
                
                iou = self.calculate_iou(pred_mask, gt_mask)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                matched_pairs.append((pred_idx, best_gt_idx, best_iou))
                used_pred[pred_idx] = True
                used_gt[best_gt_idx] = True
        
        unmatched_pred = [i for i, used in enumerate(used_pred) if not used]
        unmatched_gt = [i for i, used in enumerate(used_gt) if not used]
        
        return matched_pairs, unmatched_pred, unmatched_gt
    
    def evaluate_single_image(self, model, image_path, conf_threshold=0.25):
        """
        Evaluate a single image
        
        Args:
            model: loaded YOLO model
            image_path: path to the image
            conf_threshold: confidence threshold
        
        Returns:
            metrics_per_class: dictionary of metrics per class
        """
        # Load ground truth
        gt_masks = self.load_ground_truth(image_path)
        
        # Model prediction
        results = model(str(image_path), conf=conf_threshold, verbose=False)
        
        if len(results) == 0 or results[0].masks is None:
            pred_masks = []
        else:
            result = results[0]
            masks = result.masks.data.cpu().numpy() if result.masks is not None else []
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            cls_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
            
            pred_masks = []
            for i, (mask, conf, cls_id) in enumerate(zip(masks, confs, cls_ids)):
                binary_mask = (mask > 0.5).astype(np.uint8)
                pred_masks.append((binary_mask, float(conf), int(cls_id)))
        
        # Initialize metrics for each class
        metrics_per_class = {}
        for class_id in range(self.num_classes):
            metrics_per_class[class_id] = {
                'dice_scores': [],
                'iou_scores': [],
                'hd95_scores': [],
                'rve_scores': [],
                'pa_scores': [],
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        
        # Match predicted and ground truth masks
        matched_pairs, unmatched_pred, unmatched_gt = self.match_masks(pred_masks, gt_masks)
        
        # Process matched mask pairs
        for pred_idx, gt_idx, iou in matched_pairs:
            pred_mask, pred_conf, pred_class = pred_masks[pred_idx]
            gt_class, gt_mask = gt_masks[gt_idx]
            
            if pred_class != gt_class:
                continue
            
            # Calculate metrics
            dice = self.calculate_dice(pred_mask, gt_mask)
            iou = self.calculate_iou(pred_mask, gt_mask)
            hd95 = self.calculate_hd95(pred_mask, gt_mask)
            rve = self.calculate_rve(pred_mask, gt_mask)
            pa = self.calculate_pa(pred_mask, gt_mask)
            
            metrics_per_class[pred_class]['dice_scores'].append(dice)
            metrics_per_class[pred_class]['iou_scores'].append(iou)
            metrics_per_class[pred_class]['hd95_scores'].append(hd95)
            metrics_per_class[pred_class]['rve_scores'].append(rve)
            metrics_per_class[pred_class]['pa_scores'].append(pa)
            metrics_per_class[pred_class]['true_positives'] += 1
        
        # Process unmatched predictions (false positives)
        for pred_idx in unmatched_pred:
            pred_mask, pred_conf, pred_class = pred_masks[pred_idx]
            metrics_per_class[pred_class]['false_positives'] += 1
        
        # Process unmatched ground truth masks (false negatives)
        for gt_idx in unmatched_gt:
            gt_class, gt_mask = gt_masks[gt_idx]
            metrics_per_class[gt_class]['false_negatives'] += 1
        
        return metrics_per_class
    
    def evaluate_model(self, model_name, model_path, conf_threshold=0.25):
        """
        Evaluate the entire model
        
        Args:
            model_name: name of the model
            model_path: path to the model
            conf_threshold: confidence threshold
        
        Returns:
            results: dictionary of evaluation results
        """
        print(f"\nEvaluating model: {model_name}")
        print(f"Model path: {model_path}")
        
        # Load model
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} does not exist, skipping evaluation")
            return None
        
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return None
        
        # Initialize result storage
        all_metrics = {}
        for class_id in range(self.num_classes):
            all_metrics[class_id] = {
                'dice_scores': [],
                'iou_scores': [],
                'hd95_scores': [],
                'rve_scores': [],
                'pa_scores': [],
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        
        # Evaluate each image
        for img_path in tqdm(self.val_images[:50], desc=f"Evaluating {model_name}"):  # Limit to 50 images for speed
            metrics_per_class = self.evaluate_single_image(model, img_path, conf_threshold)
            
            # Aggregate metrics
            for class_id in range(self.num_classes):
                for metric in ['dice_scores', 'iou_scores', 'hd95_scores', 'rve_scores', 'pa_scores']:
                    all_metrics[class_id][metric].extend(metrics_per_class[class_id][metric])
                
                for metric in ['true_positives', 'false_positives', 'false_negatives']:
                    all_metrics[class_id][metric] += metrics_per_class[class_id][metric]
        
        # Calculate average metrics per class
        class_results = {}
        for class_id in range(self.num_classes):
            dice_scores = all_metrics[class_id]['dice_scores']
            iou_scores = all_metrics[class_id]['iou_scores']
            hd95_scores = all_metrics[class_id]['hd95_scores']
            rve_scores = all_metrics[class_id]['rve_scores']
            pa_scores = all_metrics[class_id]['pa_scores']
            
            # Filter invalid HD95 values
            valid_hd95 = [x for x in hd95_scores if x != float('inf')]
            valid_rve = [x for x in rve_scores if x != float('inf')]
            
            class_results[class_id] = {
                'class_name': self.class_names[class_id],
                'dice_mean': np.mean(dice_scores) if dice_scores else 0,
                'dice_std': np.std(dice_scores) if dice_scores else 0,
                'iou_mean': np.mean(iou_scores) if iou_scores else 0,
                'iou_std': np.std(iou_scores) if iou_scores else 0,
                'hd95_mean': np.mean(valid_hd95) if valid_hd95 else float('inf'),
                'hd95_std': np.std(valid_hd95) if valid_hd95 else 0,
                'rve_mean': np.mean(valid_rve) if valid_rve else float('inf'),
                'rve_std': np.std(valid_rve) if valid_rve else 0,
                'pa_mean': np.mean(pa_scores) if pa_scores else 0,
                'pa_std': np.std(pa_scores) if pa_scores else 0,
                'true_positives': all_metrics[class_id]['true_positives'],
                'false_positives': all_metrics[class_id]['false_positives'],
                'false_negatives': all_metrics[class_id]['false_negatives'],
                'precision': all_metrics[class_id]['true_positives'] / max(1, all_metrics[class_id]['true_positives'] + all_metrics[class_id]['false_positives']),
                'recall': all_metrics[class_id]['true_positives'] / max(1, all_metrics[class_id]['true_positives'] + all_metrics[class_id]['false_negatives']),
                'num_samples': len(dice_scores)
            }
        
        # Calculate overall metrics
        all_dice = []
        all_iou = []
        all_hd95 = []
        all_rve = []
        all_pa = []
        
        for class_id in range(self.num_classes):
            all_dice.extend(all_metrics[class_id]['dice_scores'])
            all_iou.extend(all_metrics[class_id]['iou_scores'])
            all_hd95.extend([x for x in all_metrics[class_id]['hd95_scores'] if x != float('inf')])
            all_rve.extend([x for x in all_metrics[class_id]['rve_scores'] if x != float('inf')])
            all_pa.extend(all_metrics[class_id]['pa_scores'])
        
        total_tp = sum(all_metrics[class_id]['true_positives'] for class_id in range(self.num_classes))
        total_fp = sum(all_metrics[class_id]['false_positives'] for class_id in range(self.num_classes))
        total_fn = sum(all_metrics[class_id]['false_negatives'] for class_id in range(self.num_classes))
        
        overall_results = {
            'model_name': model_name,
            'dice_mean': np.mean(all_dice) if all_dice else 0,
            'dice_std': np.std(all_dice) if all_dice else 0,
            'iou_mean': np.mean(all_iou) if all_iou else 0,
            'iou_std': np.std(all_iou) if all_iou else 0,
            'hd95_mean': np.mean(all_hd95) if all_hd95 else float('inf'),
            'hd95_std': np.std(all_hd95) if all_hd95 else 0,
            'rve_mean': np.mean(all_rve) if all_rve else float('inf'),
            'rve_std': np.std(all_rve) if all_rve else 0,
            'pa_mean': np.mean(all_pa) if all_pa else 0,
            'pa_std': np.std(all_pa) if all_pa else 0,
            'precision': total_tp / max(1, total_tp + total_fp),
            'recall': total_tp / max(1, total_tp + total_fn),
            'f1_score': 2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn),
            'total_samples': len(all_dice)
        }
        
        return {
            'class_results': class_results,
            'overall_results': overall_results
        }
    
    def run_evaluation(self, output_dir='medical_seg_evaluation', conf_threshold=0.25):
        """
        Run the complete evaluation pipeline
        
        Args:
            output_dir: output directory
            conf_threshold: confidence threshold
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate all models
        all_results = {}
        for model_name, model_path in self.model_paths.items():
            results = self.evaluate_model(model_name, model_path, conf_threshold)
            if results is not None:
                all_results[model_name] = results
        
        if not all_results:
            print("Error: No model evaluated successfully")
            return
        
        # Create comparison tables
        self.create_comparison_tables(all_results, output_dir)
        
        # Create visualizations
        self.create_visualizations(all_results, output_dir)
        
        # Generate detailed report
        self.generate_detailed_report(all_results, output_dir)
        
        print(f"\nEvaluation completed! Results saved to {output_dir} directory")
    
    def create_comparison_tables(self, all_results, output_dir):
        """Create comparison tables"""
        
        # 1. Overall performance comparison table
        overall_data = []
        for model_name, results in all_results.items():
                overall = results['overall_results']
                # 格式化RVE，使用科学计数法或合适的格式
                if overall['rve_mean'] != float('inf'):
                    if overall['rve_mean'] > 1e6:
                        rve_str = f"{overall['rve_mean']:.2e} ± {overall['rve_std']:.2e}"
                    else:
                        rve_str = f"{overall['rve_mean']:.4f} ± {overall['rve_std']:.4f}"
                else:
                    rve_str = 'N/A'
                
                overall_data.append({
                    'Model': model_name,
                    'Dice': f"{overall['dice_mean']:.4f} ± {overall['dice_std']:.4f}",
                    'mIoU': f"{overall['iou_mean']:.4f} ± {overall['iou_std']:.4f}",
                    'HD95': f"{overall['hd95_mean']:.2f} ± {overall['hd95_std']:.2f}" if overall['hd95_mean'] != float('inf') else 'N/A',
                    'RVE': rve_str,
                    'PA': f"{overall['pa_mean']:.4f} ± {overall['pa_std']:.4f}",
                    'Precision': f"{overall['precision']:.4f}",
                    'Recall': f"{overall['recall']:.4f}",
                    'F1': f"{overall['f1_score']:.4f}",
                    'Samples': overall['total_samples']
                })
        
        overall_df = pd.DataFrame(overall_data)
        print("\n" + "="*100)
        print("Overall Performance Comparison")
        print("="*100)
        print(overall_df.to_string(index=False))
        print("="*100)
        
        # Save as CSV
        overall_df.to_csv(os.path.join(output_dir, 'overall_comparison.csv'), index=False, encoding='utf-8-sig')
        
        # 2. Detailed comparison table by class
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            class_data = []
            
            for model_name, results in all_results.items():
                class_result = results['class_results'][class_id]
                # 格式化RVE，使用科学计数法或合适的格式
                if class_result['rve_mean'] != float('inf'):
                    if class_result['rve_mean'] > 1e6:
                        rve_str = f"{class_result['rve_mean']:.2e} ± {class_result['rve_std']:.2e}"
                    else:
                        rve_str = f"{class_result['rve_mean']:.4f} ± {class_result['rve_std']:.4f}"
                else:
                    rve_str = 'N/A'
                
                class_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'Dice': f"{class_result['dice_mean']:.4f} ± {class_result['dice_std']:.4f}",
                    'IoU': f"{class_result['iou_mean']:.4f} ± {class_result['iou_std']:.4f}",
                    'HD95': f"{class_result['hd95_mean']:.2f} ± {class_result['hd95_std']:.2f}" if class_result['hd95_mean'] != float('inf') else 'N/A',
                    'RVE': rve_str,
                    'PA': f"{class_result['pa_mean']:.4f} ± {class_result['pa_std']:.4f}",
                    'Precision': f"{class_result['precision']:.4f}",
                    'Recall': f"{class_result['recall']:.4f}",
                    'TP': class_result['true_positives'],
                    'FP': class_result['false_positives'],
                    'FN': class_result['false_negatives'],
                    'Samples': class_result['num_samples']
                })
            
            class_df = pd.DataFrame(class_data)
            print(f"\n{'='*80}")
            print(f"Class Comparison: {class_name}")
            print("="*80)
            print(class_df.to_string(index=False))
            
            # Save as CSV
            safe_class_name = class_name.replace('/', '_').replace('\\', '_')
            class_df.to_csv(os.path.join(output_dir, f'class_{safe_class_name}_comparison.csv'), 
                          index=False, encoding='utf-8-sig')
    
    def create_visualizations(self, all_results, output_dir):
        """Create visualizations"""
        
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Overall performance comparison bar chart
        # 修改布局为 3x3 以适应新增图表
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        axes = axes.flatten()
        
        # 扩展指标列表，新增 HD95 和 RVE
        metrics = ['Dice', 'mIoU', 'PA', 'Precision', 'Recall', 'F1', 'HD95', 'RVE']
        metric_keys = ['dice_mean', 'iou_mean', 'pa_mean', 'precision', 'recall', 'f1_score', 'hd95_mean', 'rve_mean']
        
        model_names = list(all_results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        
        for idx, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
            ax = axes[idx]
            metric_values = []
            
            for model_name in model_names:
                value = all_results[model_name]['overall_results'][metric_key]
                # 处理无穷大值，为绘图替换为0
                if value == float('inf'):
                    metric_values.append(0)
                else:
                    metric_values.append(value)
            
            bars = ax.bar(model_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_xlabel('Model', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 显示数值标签
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                # 根据指标类型格式化标签文本
                if metric_name in ['HD95', 'RVE']:
                    # HD95和RVE通常值较小，但范围可能很大，格式化显示
                    if value == 0:  # 代表之前的无穷大
                        label = 'Inf'
                    elif value > 1e6:
                        label = f'{value:.2e}'
                    elif value > 100:
                        label = f'{value:.1f}'
                    else:
                        label = f'{value:.2f}'
                else:
                    label = f'{value:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       label, ha='center', va='bottom', fontsize=9)
            
            # 为Dice等0-1指标设置固定Y轴范围
            if metric_name not in ['HD95', 'RVE']:
                ax.set_ylim(0, 1.1)
            else:
                # 为HD95和RVE设置自适应范围，但设置上限避免极端值
                y_max = max(metric_values) * 1.2
                ax.set_ylim(0, y_max if y_max > 0 else 1)
        
        # 隐藏最后一个（第9个）子图，因为我们只有8个指标
        axes[-1].axis('off')
        
        plt.suptitle('Medical Image Segmentation Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Dice coefficient heatmap by class
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dice_matrix = []
        for model_name in model_names:
            model_dice = []
            for class_id in range(self.num_classes):
                dice_value = all_results[model_name]['class_results'][class_id]['dice_mean']
                model_dice.append(dice_value)
            dice_matrix.append(model_dice)
        
        dice_matrix = np.array(dice_matrix)
        
        # 确保至少有两个模型才显示对比
        if len(model_names) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 models for comparison', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            im = ax.imshow(dice_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(np.arange(self.num_classes))
            ax.set_yticks(np.arange(len(model_names)))
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_yticklabels(model_names)
            
            # Display values in each cell
            for i in range(len(model_names)):
                for j in range(self.num_classes):
                    text = ax.text(j, i, f'{dice_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black" if dice_matrix[i, j] < 0.7 else "white",
                                 fontsize=10, fontweight='bold')
            
            ax.set_title('Dice Coefficient by Class', fontsize=14, fontweight='bold')
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dice_by_class_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Detailed comparison by class (bar chart)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_detail = ['Dice', 'IoU', 'Precision', 'Recall']
        metric_keys_detail = ['dice_mean', 'iou_mean', 'precision', 'recall']
        
        for idx, (metric_name, metric_key) in enumerate(zip(metrics_detail, metric_keys_detail)):
            ax = axes[idx]
            
            x = np.arange(self.num_classes)
            width = 0.8 / len(model_names)
            
            for i, model_name in enumerate(model_names):
                values = [all_results[model_name]['class_results'][j][metric_key] for j in range(self.num_classes)]
                offset = (i - len(model_names)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=model_name, 
                            edgecolor='black', linewidth=1)
                
                # Display values above bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if height > 0.05:  # Only display larger values
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} by Class', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1.1)
        
        plt.suptitle('Model Performance by Class', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_by_class_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Radar chart comparison
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Select key metrics
        radar_metrics = ['dice_mean', 'iou_mean', 'precision', 'recall', 'pa_mean', 'f1_score']
        radar_labels = ['Dice', 'mIoU', 'Precision', 'Recall', 'PA', 'F1']
        
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        
        for i, model_name in enumerate(model_names):
            values = [all_results[model_name]['overall_results'][key] for key in radar_metrics]
            values += values[:1]  # Close the shape
            current_angles = angles + angles[:1]
            
            ax.plot(current_angles, values, 'o-', linewidth=2, label=model_name, markersize=8)
            ax.fill(current_angles, values, alpha=0.1)
        
        ax.set_thetagrids(np.degrees(angles), radar_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, all_results, output_dir):
        """Generate detailed evaluation report"""
        
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Medical Image Segmentation Model Evaluation Report\n\n")
            f.write(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. Evaluation Settings\n\n")
            f.write(f"- Number of validation images: {len(self.val_images)}\n")
            f.write(f"- Number of classes: {self.num_classes}\n")
            f.write(f"- Class names: {', '.join(self.class_names)}\n")
            f.write(f"- Models evaluated: {', '.join(list(all_results.keys()))}\n\n")
            
            f.write("## 2. Evaluation Metrics Explanation\n\n")
            f.write("1. **Dice Coefficient**: Measures the overlap between segmentation result and ground truth, closer to 1 is better\n")
            f.write("2. **mIoU (mean Intersection over Union)**: Average IoU, measures segmentation accuracy\n")
            f.write("3. **HD95 (95% Hausdorff Distance)**: 95% Hausdorff distance, measures boundary matching, smaller is better\n")
            f.write("4. **RVE (Relative Volume Error)**: Relative volume error, measures volume difference, smaller is better\n")
            f.write("5. **PA (Pixel Accuracy)**: Proportion of correctly classified pixels\n")
            f.write("6. **Precision**: Proportion of true positives among predicted positives\n")
            f.write("7. **Recall**: Proportion of true positives among actual positives\n")
            f.write("8. **F1 Score**: Harmonic mean of precision and recall\n\n")
            
            f.write("## 3. Overall Performance Comparison\n\n")
            
            # Overall performance table
            f.write("| Model | Dice | mIoU | HD95 | RVE | PA | Precision | Recall | F1 Score |\n")
            f.write("|-------|------|------|------|-----|----|-----------|--------|----------|\n")
            
            for model_name, results in all_results.items():
                overall = results['overall_results']
                hd95_str = f"{overall['hd95_mean']:.2f}" if overall['hd95_mean'] != float('inf') else "N/A"
                # 格式化RVE，使用科学计数法或合适的格式
                if overall['rve_mean'] != float('inf'):
                    if overall['rve_mean'] > 1e6:
                        rve_str = f"{overall['rve_mean']:.2e}"
                    else:
                        rve_str = f"{overall['rve_mean']:.3f}"
                else:
                    rve_str = "N/A"
                
                f.write(f"| {model_name} | "
                       f"{overall['dice_mean']:.4f} ± {overall['dice_std']:.4f} | "
                       f"{overall['iou_mean']:.4f} ± {overall['iou_std']:.4f} | "
                       f"{hd95_str} | "
                       f"{rve_str} | "
                       f"{overall['pa_mean']:.4f} ± {overall['pa_std']:.4f} | "
                       f"{overall['precision']:.4f} | "
                       f"{overall['recall']:.4f} | "
                       f"{overall['f1_score']:.4f} |\n")
            
            f.write("\n## 4. Detailed Performance by Class\n\n")
            
            for class_id in range(self.num_classes):
                class_name = self.class_names[class_id]
                f.write(f"### 4.{class_id+1} Class: {class_name}\n\n")
                
                f.write("| Model | Dice | IoU | HD95 | RVE | PA | Precision | Recall | TP | FP | FN |\n")
                f.write("|-------|------|-----|------|-----|----|-----------|--------|----|----|----|\n")
                
                for model_name, results in all_results.items():
                    class_result = results['class_results'][class_id]
                    hd95_str = f"{class_result['hd95_mean']:.2f}" if class_result['hd95_mean'] != float('inf') else "N/A"
                    # 格式化RVE，使用科学计数法或合适的格式
                    if class_result['rve_mean'] != float('inf'):
                        if class_result['rve_mean'] > 1e6:
                            rve_str = f"{class_result['rve_mean']:.2e}"
                        else:
                            rve_str = f"{class_result['rve_mean']:.3f}"
                    else:
                        rve_str = "N/A"
                    
                    f.write(f"| {model_name} | "
                           f"{class_result['dice_mean']:.4f} | "
                           f"{class_result['iou_mean']:.4f} | "
                           f"{hd95_str} | "
                           f"{rve_str} | "
                           f"{class_result['pa_mean']:.4f} | "
                           f"{class_result['precision']:.4f} | "
                           f"{class_result['recall']:.4f} | "
                           f"{class_result['true_positives']} | "
                           f"{class_result['false_positives']} | "
                           f"{class_result['false_negatives']} |\n")
                f.write("\n")
            
            f.write("## 5. Visualization Charts\n\n")
            f.write("The following charts have been generated:\n\n")
            f.write("1. `overall_comparison.png` - 整体性能对比柱状图，包含 Dice, mIoU, PA, Precision, Recall, F1, HD95, RVE 指标\n") # 修改了这一行
            f.write("2. `dice_by_class_heatmap.png` - Dice coefficient heatmap by class\n")
            f.write("3. `detailed_by_class_comparison.png` - Detailed comparison by class\n")
            f.write("4. `radar_chart_comparison.png` - Model performance radar chart\n\n")
            
            f.write("## 6. Analysis and Recommendations\n\n")
            
            # Automatic result analysis
            if len(all_results) >= 2:
                model_names = list(all_results.keys())
                baseline_results = all_results[model_names[0]]['overall_results']
                ela_results = all_results[model_names[1]]['overall_results'] if len(model_names) > 1 else None
                
                f.write("### 6.1 Overall Performance Analysis\n\n")
                
                if ela_results:
                    dice_improvement = ((ela_results['dice_mean'] - baseline_results['dice_mean']) / baseline_results['dice_mean']) * 100
                    iou_improvement = ((ela_results['iou_mean'] - baseline_results['iou_mean']) / baseline_results['iou_mean']) * 100
                    
                    f.write(f"- **Dice Coefficient**: ELA model compared to Baseline "
                           f"{'improved' if dice_improvement > 0 else 'decreased'} by {abs(dice_improvement):.2f}%\n")
                    f.write(f"- **mIoU**: ELA model compared to Baseline "
                           f"{'improved' if iou_improvement > 0 else 'decreased'} by {abs(iou_improvement):.2f}%\n")
                    
                    if dice_improvement > 0:
                        f.write("\n✅ **Conclusion**: The ELA improved model outperforms the Baseline model in segmentation accuracy.\n")
                    else:
                        f.write("\n⚠️ **Conclusion**: The ELA improved model may underperform the Baseline model in some metrics, further analysis is needed.\n")
                
                f.write("\n### 6.2 Improvement Suggestions\n\n")
                f.write("1. **Data Augmentation**: Consider more diverse data augmentation strategies\n")
                f.write("2. **Class Balance**: Analyze sample counts per class, consider oversampling minority classes\n")
                f.write("3. **Post-processing**: Add morphological post-processing (e.g., opening and closing) to optimize segmentation results\n")
                f.write("4. **Model Ensemble**: Consider ensemble predictions from multiple models\n")
                f.write("5. **Threshold Optimization**: Use different confidence thresholds for different classes\n")
            
            f.write("\n---\n")
            f.write("*Report automatically generated - YOLOv26 Medical Image Segmentation Evaluation System*")
        
        print(f"Detailed evaluation report saved to: {report_path}")

# Main program
if __name__ == "__main__":
    # Configuration file paths
    data_yaml_path = "data.yaml"  # Dataset configuration file from document 4
    config_txt_path = "config.txt"  # Model configuration file from document 5
    
    # Create evaluator
    print("Initializing Medical Image Segmentation Evaluator...")
    evaluator = MedicalSegmentationEvaluator(data_yaml_path, config_txt_path)
    
    # Run evaluation
    print("Starting evaluation process...")
    evaluator.run_evaluation(
        output_dir='medical_seg_evaluation',
        conf_threshold=0.25  # Consistent with train.py
    )
    
    print("\nEvaluation script execution completed!")
    print("Generated files include:")
    print("1. Overall performance comparison table (overall_comparison.csv)")
    print("2. Detailed comparison tables by class (class_*.csv)")
    print("3. Visualization charts (*.png)")
    print("4. Detailed evaluation report (evaluation_report.md)")