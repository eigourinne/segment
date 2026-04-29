import streamlit as st
import os
import tempfile
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import pandas as pd
from datetime import datetime
import json
import zipfile
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')  # 设置matplotlib为非交互式后端

# ------------------ 页面配置 ------------------
st.set_page_config(
    page_title="YOLOv26 医学图像分割与量化分析系统",
    page_icon="🩺",
    layout="wide"
)

# ------------------ 会话状态初始化 ------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = ""
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = {}
if 'quantitative_results' not in st.session_state:
    st.session_state.quantitative_results = {}
if 'case_studies' not in st.session_state:
    st.session_state.case_studies = {}
if 'current_case_id' not in st.session_state:
    st.session_state.current_case_id = None
if 'conf_threshold' not in st.session_state:
    st.session_state.conf_threshold = 0.25
if 'pixel_size_mm' not in st.session_state:
    st.session_state.pixel_size_mm = 0.1  # 默认像素尺寸，单位：mm

# ------------------ 辅助函数 ------------------
def auto_load_model():
    """自动查找并加载模型"""
    possible_paths = [
        "runs/segment/train/weights/best.pt",
        "./best.pt",
        "best.pt"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                st.session_state.model = model
                st.session_state.model_path = model_path
                st.success(f"✅ 模型加载成功: `{model_path}`")
                return True
            except Exception as e:
                st.error(f"加载模型 `{model_path}` 时出错: {e}")
                return False
    
    st.warning("⚠️ 未在默认路径找到模型文件 'best.pt'，请通过侧边栏手动上传。")
    return False

def calculate_morphological_params(mask, pixel_size_mm=0.1):
    """
    计算病灶区域的形态学参数
    
    参数:
        mask: 二值掩码图像 (numpy数组)
        pixel_size_mm: 像素对应的物理尺寸 (mm)
    
    返回:
        包含形态学参数的字典
    """
    if mask is None or np.sum(mask) == 0:
        return None
    
    # 确保掩码是二值的
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 找到所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    params_list = []
    
    # 遍历所有连通区域（跳过背景，索引0）
    for label in range(1, num_labels):
        # 提取当前区域的掩码
        region_mask = (labels == label).astype(np.uint8)
        
        # 1. 面积（像素数和物理面积）
        pixel_area = np.sum(region_mask)
        physical_area_mm2 = pixel_area * (pixel_size_mm ** 2)
        
        # 2. 周长
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter_pixels = cv2.arcLength(contours[0], True)
            perimeter_mm = perimeter_pixels * pixel_size_mm
        else:
            perimeter_pixels = 0
            perimeter_mm = 0
        
        # 3. 边界框
        y_indices, x_indices = np.where(region_mask > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox_width = (x_max - x_min + 1) * pixel_size_mm
            bbox_height = (y_max - y_min + 1) * pixel_size_mm
            bbox_area = bbox_width * bbox_height
        else:
            x_min = x_max = y_min = y_max = bbox_width = bbox_height = bbox_area = 0
        
        # 4. 质心坐标
        if centroids is not None and label < len(centroids):
            centroid_x, centroid_y = centroids[label]
            centroid_x_mm = centroid_x * pixel_size_mm
            centroid_y_mm = centroid_y * pixel_size_mm
        else:
            M = cv2.moments(region_mask)
            if M["m00"] != 0:
                centroid_x = M["m10"] / M["m00"]
                centroid_y = M["m01"] / M["m00"]
                centroid_x_mm = centroid_x * pixel_size_mm
                centroid_y_mm = centroid_y * pixel_size_mm
            else:
                centroid_x = centroid_y = centroid_x_mm = centroid_y_mm = 0
        
        # 5. 最大直径 (Feret直径)
        if len(contours) > 0:
            # 获取轮廓点
            contour_points = contours[0].reshape(-1, 2)
            max_distance = 0
            max_points = None
            
            # 计算所有点对之间的最大距离
            for i in range(len(contour_points)):
                for j in range(i+1, len(contour_points)):
                    distance = np.linalg.norm(contour_points[i] - contour_points[j])
                    if distance > max_distance:
                        max_distance = distance
                        max_points = (contour_points[i], contour_points[j])
            
            max_diameter_pixels = max_distance
            max_diameter_mm = max_diameter_pixels * pixel_size_mm
        else:
            max_diameter_pixels = 0
            max_diameter_mm = 0
            max_points = None
        
        # 6. 圆形度
        if perimeter_pixels > 0:
            circularity = (4 * np.pi * pixel_area) / (perimeter_pixels ** 2)
        else:
            circularity = 0
        
        # 7. 伸长度 (Aspect Ratio)
        if bbox_height > 0:
            aspect_ratio = bbox_width / bbox_height
        else:
            aspect_ratio = 0
        
        # 8. 紧凑度
        if pixel_area > 0:
            compactness = (perimeter_pixels ** 2) / (4 * np.pi * pixel_area)
        else:
            compactness = 0
        
        params = {
            "region_id": label,
            "pixel_area": int(pixel_area),
            "physical_area_mm2": round(physical_area_mm2, 2),
            "perimeter_pixels": round(perimeter_pixels, 2),
            "perimeter_mm": round(perimeter_mm, 2),
            "centroid": (round(centroid_x, 2), round(centroid_y, 2)),
            "centroid_mm": (round(centroid_x_mm, 2), round(centroid_y_mm, 2)),
            "bbox": {
                "x_min": int(x_min), "x_max": int(x_max),
                "y_min": int(y_min), "y_max": int(y_max),
                "width_pixels": int(x_max - x_min + 1),
                "height_pixels": int(y_max - y_min + 1),
                "width_mm": round(bbox_width, 2),
                "height_mm": round(bbox_height, 2),
                "area_mm2": round(bbox_area, 2)
            },
            "max_diameter_pixels": round(max_diameter_pixels, 2),
            "max_diameter_mm": round(max_diameter_mm, 2),
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "compactness": round(compactness, 3)
        }
        
        params_list.append(params)
    
    return params_list

def predict_and_analyze(img_path, conf_threshold=0.25, pixel_size_mm=0.1):
    """对单张图片进行预测并返回可视化结果和量化分析"""
    if st.session_state.model is None:
        raise ValueError("模型未加载，无法进行预测。")
    
    # 读取原图
    original_img = cv2.imread(img_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 使用模型进行预测
    results = st.session_state.model(img_path, conf=conf_threshold)
    
    # 处理预测结果
    has_detection = False
    detections = []
    quantitative_data = []
    all_masks = None
    
    if len(results) > 0 and results[0].masks is not None:
        result = results[0]
        plotted_img = result.plot(masks=True, boxes=True, labels=True)
        result_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        has_detection = True
        
        # 提取掩码
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        
        # 创建合并的掩码用于显示
        if masks is not None and len(masks) > 0:
            all_masks = np.zeros_like(masks[0], dtype=np.uint8)
            for i, mask in enumerate(masks):
                # 将掩码转换为二值图像
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                all_masks = cv2.bitwise_or(all_masks, binary_mask)
        
        # 提取检测信息和量化分析
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                class_name = result.names[cls_id] if hasattr(result, 'names') else f"Class {cls_id}"
                
                # 提取当前目标的掩码
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    # 计算形态学参数
                    morph_params = calculate_morphological_params(binary_mask, pixel_size_mm)
                else:
                    morph_params = None
                
                detections.append({
                    "id": i + 1,
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": class_name
                })
                
                if morph_params:
                    for region_params in morph_params:
                        quantitative_data.append({
                            "detection_id": i + 1,
                            "class_name": class_name,
                            "confidence": float(conf),
                            **region_params
                        })
    else:
        result_img_rgb = original_img_rgb.copy()
        masks = None
    
    return original_img_rgb, result_img_rgb, has_detection, detections, quantitative_data, all_masks

def generate_quantitative_report(quantitative_data, img_name):
    """生成量化分析报告"""
    if not quantitative_data:
        return None
    
    report = {
        "image_name": img_name,
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_regions": len(quantitative_data),
        "detections_by_class": {},
        "summary_statistics": {},
        "detailed_measurements": quantitative_data
    }
    
    # 按类别统计
    for data in quantitative_data:
        class_name = data["class_name"]
        if class_name not in report["detections_by_class"]:
            report["detections_by_class"][class_name] = 0
        report["detections_by_class"][class_name] += 1
    
    # 计算汇总统计
    if quantitative_data:
        areas = [d["physical_area_mm2"] for d in quantitative_data]
        perimeters = [d["perimeter_mm"] for d in quantitative_data]
        diameters = [d["max_diameter_mm"] for d in quantitative_data]
        
        report["summary_statistics"] = {
            "area_mm2": {
                "mean": round(np.mean(areas), 2),
                "std": round(np.std(areas), 2),
                "min": round(min(areas), 2),
                "max": round(max(areas), 2),
                "total": round(sum(areas), 2)
            },
            "perimeter_mm": {
                "mean": round(np.mean(perimeters), 2),
                "std": round(np.std(perimeters), 2),
                "min": round(min(perimeters), 2),
                "max": round(max(perimeters), 2)
            },
            "diameter_mm": {
                "mean": round(np.mean(diameters), 2),
                "std": round(np.std(diameters), 2),
                "min": round(min(diameters), 2),
                "max": round(max(diameters), 2)
            }
        }
    
    return report

def create_comparison_chart(case_data):
    """创建病例对比分析图表"""
    if not case_data or len(case_data) < 2:
        return None
    
    # 准备数据
    dates = sorted(case_data.keys())
    
    # 提取每个时间点的数据
    areas = []
    diameters = []
    region_counts = []
    
    for date in dates:
        data = case_data[date]
        if data["quantitative_data"]:
            total_area = sum([d["physical_area_mm2"] for d in data["quantitative_data"]])
            avg_diameter = np.mean([d["max_diameter_mm"] for d in data["quantitative_data"]])
            areas.append(total_area)
            diameters.append(avg_diameter)
            region_counts.append(len(data["quantitative_data"]))
        else:
            areas.append(0)
            diameters.append(0)
            region_counts.append(0)
    
    # 创建图表
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('病灶总面积变化趋势', '平均最大直径变化趋势', 
                       '病灶数量变化趋势', '形态参数对比'),
        vertical_spacing=0.15
    )
    
    # 总面积趋势
    fig.add_trace(
        go.Scatter(x=dates, y=areas, mode='lines+markers', name='总面积(mm²)',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # 直径趋势
    fig.add_trace(
        go.Scatter(x=dates, y=diameters, mode='lines+markers', name='平均直径(mm)',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # 数量趋势
    fig.add_trace(
        go.Bar(x=dates, y=region_counts, name='病灶数量',
              marker_color='green'),
        row=2, col=1
    )
    
    # 形态参数对比（最后一次检查）
    if case_data[dates[-1]]["quantitative_data"]:
        latest_data = case_data[dates[-1]]["quantitative_data"][0]  # 取第一个区域
        params = ['circularity', 'aspect_ratio', 'compactness']
        values = [latest_data[p] for p in params]
        
        fig.add_trace(
            go.Bar(x=params, y=values, name='形态参数',
                  marker_color=['orange', 'purple', 'brown']),
            row=2, col=2
        )
    
    # 更新布局
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text=f"病例对比分析 ({len(dates)} 次检查)",
        title_font_size=16
    )
    
    fig.update_xaxes(title_text="检查日期", row=1, col=1)
    fig.update_xaxes(title_text="检查日期", row=1, col=2)
    fig.update_xaxes(title_text="检查日期", row=2, col=1)
    fig.update_xaxes(title_text="参数类型", row=2, col=2)
    
    fig.update_yaxes(title_text="总面积 (mm²)", row=1, col=1)
    fig.update_yaxes(title_text="平均直径 (mm)", row=1, col=2)
    fig.update_yaxes(title_text="数量", row=2, col=1)
    fig.update_yaxes(title_text="数值", row=2, col=2)
    
    return fig

def generate_pdf_report(case_data, case_id):
    """生成PDF报告"""
    if not case_data:
        return None
    
    # 创建缓冲区
    buffer = BytesIO()
    
    # 创建PDF文档
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    # 获取样式
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # 自定义样式
    small_style = ParagraphStyle(
        'Small',
        parent=normal_style,
        fontSize=8,
        leading=10
    )
    
    # 构建报告内容
    story = []
    
    # 标题 (修改为英文)
    story.append(Paragraph(f"Medical Image Segmentation Analysis Report - Case {case_id}", title_style))
    story.append(Spacer(1, 12))
    
    # 基本信息 (修改为英文)
    story.append(Paragraph(f"Report Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Paragraph(f"Number of Examinations: {len(case_data)}", normal_style))
    story.append(Spacer(1, 12))
    
    # 按检查日期添加内容
    for i, (date, data) in enumerate(sorted(case_data.items())):
        story.append(Paragraph(f"Examination {i+1}: {date}", heading_style))
        story.append(Spacer(1, 6))
        
        if data["quantitative_report"]:
            report = data["quantitative_report"]
            
            # 图像信息 (修改为英文)
            story.append(Paragraph(f"Image: {report['image_name']}", normal_style))
            story.append(Paragraph(f"Analysis Timestamp: {report['analysis_timestamp']}", normal_style))
            story.append(Paragraph(f"Total Regions Detected: {report['total_regions']}", normal_style))
            
            # 类别统计 (修改为英文)
            class_text = "Classes Detected: " + ", ".join([f"{k}({v})" for k, v in report['detections_by_class'].items()])
            story.append(Paragraph(class_text, normal_style))
            story.append(Spacer(1, 6))
            
            # 汇总统计表格 (表头和内容修改为英文)
            summary_data = report['summary_statistics']
            if summary_data:
                summary_table_data = [
                    ["Parameter", "Mean", "Std Dev", "Min", "Max", "Total"],  # 表头
                    ["Area (mm²)", 
                     f"{summary_data['area_mm2']['mean']}", 
                     f"{summary_data['area_mm2']['std']}", 
                     f"{summary_data['area_mm2']['min']}", 
                     f"{summary_data['area_mm2']['max']}", 
                     f"{summary_data['area_mm2']['total']}"],
                    ["Perimeter (mm)", 
                     f"{summary_data['perimeter_mm']['mean']}", 
                     f"{summary_data['perimeter_mm']['std']}", 
                     f"{summary_data['perimeter_mm']['min']}", 
                     f"{summary_data['perimeter_mm']['max']}", 
                     "-"],
                    ["Diameter (mm)", 
                     f"{summary_data['diameter_mm']['mean']}", 
                     f"{summary_data['diameter_mm']['std']}", 
                     f"{summary_data['diameter_mm']['min']}", 
                     f"{summary_data['diameter_mm']['max']}", 
                     "-"]
                ]
                
                summary_table = Table(summary_table_data, colWidths=[2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 12))
            
            # 详细测量数据 (修改为英文)
            if report['detailed_measurements']:
                story.append(Paragraph("Detailed Measurements:", normal_style))
                
                # 只显示前5个区域
                display_data = report['detailed_measurements'][:5]
                # 表头改为英文
                detailed_table_data = [["Region ID", "Class", "Confidence", "Area(mm²)", "Perimeter(mm)", "Diameter(mm)", "Circularity"]]
                
                for item in display_data:
                    detailed_table_data.append([
                        f"{item['region_id']}",
                        item['class_name'],
                        f"{item['confidence']:.2%}",
                        f"{item['physical_area_mm2']}",
                        f"{item['perimeter_mm']}",
                        f"{item['max_diameter_mm']}",
                        f"{item['circularity']}"
                    ])
                
                detailed_table = Table(detailed_table_data, colWidths=[1.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])
                detailed_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 0), (-1, -1), 8)
                ]))
                story.append(detailed_table)
                # 脚注改为英文
                if len(report['detailed_measurements']) > 5:
                    story.append(Paragraph(f"*Note: Showing first 5 of {len(report['detailed_measurements'])} regions.", small_style))
        
        story.append(Spacer(1, 20))
    
    # 生成PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer

# ------------------ 登录页面 ------------------
def login_page():
    st.title("🔐 YOLOv26 医学分割系统 - 管理员登录")
    
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submit = st.form_submit_button("登录")
    
    if submit:
        if username == "admin" and password == "password":
            st.session_state.authenticated = True
            st.success("登录成功！正在跳转...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("用户名或密码错误，请重试。")

# ------------------ 主应用页面 ------------------
def main_app():
    st.sidebar.title("⚙️ 控制面板")
    
    # 1. 模型管理
    st.sidebar.header("1. 模型管理")
    if st.session_state.model is None:
        if st.sidebar.button("自动加载模型", help="尝试从默认路径加载模型"):
            auto_load_model()
    
    uploaded_model = st.sidebar.file_uploader(
        "或手动上传模型文件",
        type=['pt'],
        help="上传训练好的 YOLO `.pt` 模型文件"
    )
    
    if uploaded_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            model_path = tmp_file.name
        
        try:
            model = YOLO(model_path)
            st.session_state.model = model
            st.session_state.model_path = uploaded_model.name
            st.sidebar.success(f"✅ 已加载上传的模型: `{uploaded_model.name}`")
        except Exception as e:
            st.sidebar.error(f"加载上传的模型失败: {e}")
    
    if st.session_state.model is not None:
        st.sidebar.info(f"**当前模型:** `{st.session_state.model_path}`")
    
    # 2. 参数设置
    st.sidebar.header("2. 检测参数")
    st.session_state.conf_threshold = st.sidebar.slider(
        "置信度阈值", 0.0, 1.0, st.session_state.conf_threshold, 0.05,
        help="置信度低于此值的检测结果将被忽略。"
    )
    
    st.session_state.pixel_size_mm = st.sidebar.number_input(
        "像素尺寸 (mm)", 
        min_value=0.001, 
        max_value=1.0, 
        value=st.session_state.pixel_size_mm, 
        step=0.001,
        format="%.3f",
        help="每个像素对应的实际物理尺寸（毫米），用于计算实际物理尺寸。"
    )
    
    # 3. 病例管理
    st.sidebar.header("3. 病例管理")
    case_id = st.sidebar.text_input("病例ID", placeholder="输入病例编号")
    
    col_case1, col_case2 = st.sidebar.columns(2)
    with col_case1:
        if st.sidebar.button("新建病例", key="new_case"):
            if case_id:
                if case_id not in st.session_state.case_studies:
                    st.session_state.case_studies[case_id] = {}
                    st.session_state.current_case_id = case_id
                    st.sidebar.success(f"已创建病例: {case_id}")
                else:
                    st.sidebar.warning(f"病例 {case_id} 已存在")
            else:
                st.sidebar.warning("请输入病例ID")
    
    with col_case2:
        if st.sidebar.button("加载病例", key="load_case"):
            if case_id and case_id in st.session_state.case_studies:
                st.session_state.current_case_id = case_id
                st.sidebar.success(f"已加载病例: {case_id}")
            else:
                st.sidebar.warning(f"病例 {case_id} 不存在")
    
    # 显示当前病例
    if st.session_state.current_case_id:
        st.sidebar.info(f"**当前病例:** {st.session_state.current_case_id}")
        st.sidebar.write(f"检查次数: {len(st.session_state.case_studies.get(st.session_state.current_case_id, {}))}")
    
    # 4. 图片上传
    st.sidebar.header("4. 图片上传")
    upload_option = st.sidebar.radio(
        "选择上传方式",
        ["单张图片", "多张图片", "整个文件夹 (ZIP)"]
    )
    
    if upload_option == "单张图片":
        uploaded_file = st.sidebar.file_uploader(
            "选择一张医学图片",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
            key="single_upload"
        )
        if uploaded_file is not None:
            st.session_state.uploaded_images = [uploaded_file]
    
    elif upload_option == "多张图片":
        uploaded_files = st.sidebar.file_uploader(
            "选择多张医学图片",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
            accept_multiple_files=True,
            key="multi_upload"
        )
        if uploaded_files:
            st.session_state.uploaded_images = list(uploaded_files)
    
    elif upload_option == "整个文件夹 (ZIP)":
        zip_file = st.sidebar.file_uploader(
            "上传包含图片的ZIP文件夹",
            type=['zip'],
            key="zip_upload"
        )
        if zip_file is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(tmp_dir)
                
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(Path(tmp_dir).rglob(f"*{ext}"))
                
                if image_paths:
                    # 确保使用全局的BytesIO
                    uploaded_images = []
                    for img_path in image_paths:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                        file_like_object = BytesIO(img_bytes)  # 使用全局BytesIO
                        file_like_object.name = img_path.name
                        uploaded_images.append(file_like_object)
                    
                    st.session_state.uploaded_images = uploaded_images
                    st.sidebar.success(f"从ZIP文件中找到 {len(uploaded_images)} 张图片。")
                else:
                    st.sidebar.warning("ZIP文件中未找到支持的图片格式。")
    
    # 清空按钮
    if st.sidebar.button("清空图片列表"):
        st.session_state.uploaded_images = []
        st.session_state.processed_results = {}
        st.session_state.quantitative_results = {}
        st.sidebar.success("已清空图片列表。")
    
    # 5. 处理按钮
    st.sidebar.header("5. 执行处理")
    process_button = st.sidebar.button(
        "🚀 开始分割处理",
        type="primary",
        disabled=(st.session_state.model is None or len(st.session_state.uploaded_images) == 0),
        help="加载模型并选择图片后即可使用。"
    )
    
    # 主界面
    st.title("🩺 YOLOv26 医学分割与量化分析系统")
    
    # 状态显示
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        model_status = "✅ 已加载" if st.session_state.model is not None else "❌ 未加载"
        st.metric("模型状态", model_status, delta=st.session_state.model_path if st.session_state.model_path else "N/A")
    with col_status2:
        st.metric("待处理图片", len(st.session_state.uploaded_images))
    with col_status3:
        case_info = f"{st.session_state.current_case_id}" if st.session_state.current_case_id else "未选择"
        st.metric("当前病例", case_info)
    
    st.divider()
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📸 分割处理", "📊 量化分析", "📈 对比分析"])
    
    # 标签页1: 分割处理
    with tab1:
        if len(st.session_state.uploaded_images) > 0:
            st.subheader(f"📸 已选择图片 ({len(st.session_state.uploaded_images)} 张)")
            
            cols = st.columns(4)
            for idx, img_file in enumerate(st.session_state.uploaded_images):
                with cols[idx % 4]:
                    image = Image.open(img_file)
                    st.image(image, caption=img_file.name, width=200)
                    st.caption(f"{idx+1}. {img_file.name}")
            
            if process_button:
                st.subheader("🔄 处理进度与结果")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, img_file in enumerate(st.session_state.uploaded_images):
                    status_text.text(f"正在处理: {img_file.name} ({idx+1}/{len(st.session_state.uploaded_images)})")
                    progress_bar.progress((idx) / len(st.session_state.uploaded_images))
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(img_file.name).suffix) as tmp_file:
                        tmp_file.write(img_file.getbuffer())
                        tmp_img_path = tmp_file.name
                    
                    try:
                        original_img, result_img, has_detection, detections, quantitative_data, mask_img = predict_and_analyze(
                            tmp_img_path, 
                            conf_threshold=st.session_state.conf_threshold,
                            pixel_size_mm=st.session_state.pixel_size_mm
                        )
                        
                        # 生成量化报告
                        quantitative_report = generate_quantitative_report(quantitative_data, img_file.name)
                        
                        # 保存结果到会话状态
                        st.session_state.processed_results[img_file.name] = {
                            "original": original_img,
                            "result": result_img,
                            "has_detection": has_detection,
                            "detections": detections,
                            "mask": mask_img,
                            "tmp_path": tmp_img_path
                        }
                        
                        st.session_state.quantitative_results[img_file.name] = {
                            "quantitative_data": quantitative_data,
                            "quantitative_report": quantitative_report
                        }
                        
                        # 保存到当前病例
                        if st.session_state.current_case_id:
                            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if st.session_state.current_case_id not in st.session_state.case_studies:
                                st.session_state.case_studies[st.session_state.current_case_id] = {}
                            
                            st.session_state.case_studies[st.session_state.current_case_id][current_date] = {
                                "image_name": img_file.name,
                                "original_image": original_img,
                                "result_image": result_img,
                                "quantitative_data": quantitative_data,
                                "quantitative_report": quantitative_report
                            }
                        
                    except Exception as e:
                        st.error(f"处理图片 `{img_file.name}` 时出错: {e}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ 所有图片处理完成！")
                st.balloons()
        
        # 显示处理结果
        if st.session_state.processed_results:
            st.divider()
            st.subheader("📊 分割结果对比")
            
            for img_name, result_info in st.session_state.processed_results.items():
                with st.expander(f"查看详情: **{img_name}**", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**原图**")
                        st.image(result_info["original"], width=400, clamp=True)
                    
                    with col2:
                        detection_status = "✅ 检测到病变" if result_info["has_detection"] else "⚠️ 未检测到病变"
                        st.markdown(f"**分割结果** ({detection_status})")
                        st.image(result_info["result"], width=400, clamp=True)
                    
                    # 显示掩码图像
                    if result_info.get("mask") is not None:
                        st.markdown("**分割掩码**")
                        st.image(result_info["mask"], width=400, clamp=True, caption="二值掩码图像")
                    
                    # 显示检测参数与详情
                    st.markdown("**🔍 检测详情**")
                    if result_info["has_detection"] and result_info["detections"]:
                        detections_df = []
                        for det in result_info["detections"]:
                            detections_df.append({
                                "目标ID": det["id"],
                                "类别": det["class_name"],
                                "置信度": f"{det['confidence']:.2%}",
                                "边界框 (x1,y1,x2,y2)": str(det["bbox"])
                            })
                        st.table(detections_df)
                    else:
                        st.info("在此图片中未识别到医学目标。")
                    
                    # 提供下载结果图片的选项
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        result_pil = Image.fromarray(result_info["result"])
                        buf = BytesIO()  # 这里使用全局BytesIO
                        result_pil.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="⬇️ 下载分割结果图",
                            data=byte_im,
                            file_name=f"seg_result_{img_name}.png",
                            mime="image/png",
                            key=f"dl_result_{img_name}"
                        )
                    with col_dl2:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].imshow(result_info["original"])
                        axes[0].set_title("Original Image")
                        axes[0].axis('off')
                        
                        axes[1].imshow(result_info["result"])
                        status = "Detected" if result_info["has_detection"] else "Not Detected"
                        axes[1].set_title(f"Segmentation Result ({status})")
                        axes[1].axis('off')
                        
                        plt.tight_layout()
                        
                        buf_comparison = BytesIO()  # 这里使用全局BytesIO
                        plt.savefig(buf_comparison, format='png', dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        buf_comparison.seek(0)
                        
                        st.download_button(
                            label="⬇️ 下载对比图 (PNG)",
                            data=buf_comparison,
                            file_name=f"comparison_{img_name}.png",
                            mime="image/png",
                            key=f"dl_comparison_{img_name}"
                        )
                st.divider()
    
    # 标签页2: 量化分析
    with tab2:
        st.header("📊 量化分析结果")
        
        if not st.session_state.quantitative_results:
            st.info("请先上传图片并进行分割处理以查看量化分析结果。")
        else:
            for img_name, quant_info in st.session_state.quantitative_results.items():
                with st.expander(f"量化分析: {img_name}", expanded=True):
                    if quant_info["quantitative_report"]:
                        report = quant_info["quantitative_report"]
                        
                        # 显示报告摘要
                        st.subheader("分析报告摘要")
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        with col_sum1:
                            st.metric("检测区域总数", report["total_regions"])
                        with col_sum2:
                            class_info = ", ".join([f"{k}({v})" for k, v in report["detections_by_class"].items()])
                            st.metric("检测类别", class_info[:20] + "..." if len(class_info) > 20 else class_info)
                        with col_sum3:
                            st.metric("分析时间", report["analysis_timestamp"])
                        
                        # 详细测量数据表格
                        st.subheader("详细测量数据")
                        if quant_info["quantitative_data"]:
                            # 创建DataFrame
                            df_data = []
                            for item in quant_info["quantitative_data"]:
                                df_data.append({
                                    "区域ID": item["region_id"],
                                    "类别": item["class_name"],
                                    "置信度": f"{item['confidence']:.2%}",
                                    "面积(mm²)": item["physical_area_mm2"],
                                    "周长(mm)": item["perimeter_mm"],
                                    "最大直径(mm)": item["max_diameter_mm"],
                                    "质心坐标(mm)": f"({item['centroid_mm'][0]}, {item['centroid_mm'][1]})",
                                    "圆形度": item["circularity"],
                                    "伸长度": item["aspect_ratio"],
                                    "紧凑度": item["compactness"]
                                })
                            
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # 统计摘要
                            st.subheader("统计摘要")
                            if report["summary_statistics"]:
                                stats = report["summary_statistics"]
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                
                                with col_stat1:
                                    st.markdown("**面积统计 (mm²)**")
                                    st.write(f"平均值: {stats['area_mm2']['mean']}")
                                    st.write(f"标准差: {stats['area_mm2']['std']}")
                                    st.write(f"最小值: {stats['area_mm2']['min']}")
                                    st.write(f"最大值: {stats['area_mm2']['max']}")
                                    st.write(f"总计: {stats['area_mm2']['total']}")
                                
                                with col_stat2:
                                    st.markdown("**周长统计 (mm)**")
                                    st.write(f"平均值: {stats['perimeter_mm']['mean']}")
                                    st.write(f"标准差: {stats['perimeter_mm']['std']}")
                                    st.write(f"最小值: {stats['perimeter_mm']['min']}")
                                    st.write(f"最大值: {stats['perimeter_mm']['max']}")
                                
                                with col_stat3:
                                    st.markdown("**直径统计 (mm)**")
                                    st.write(f"平均值: {stats['diameter_mm']['mean']}")
                                    st.write(f"标准差: {stats['diameter_mm']['std']}")
                                    st.write(f"最小值: {stats['diameter_mm']['min']}")
                                    st.write(f"最大值: {stats['diameter_mm']['max']}")
                            
                            # 可视化图表
                            st.subheader("可视化分析")
                            if len(quant_info["quantitative_data"]) > 0:
                                fig_col1, fig_col2 = st.columns(2)
                                
                                with fig_col1:
                                    # 面积分布
                                    areas = [d["physical_area_mm2"] for d in quant_info["quantitative_data"]]
                                    fig1 = go.Figure(data=[go.Histogram(x=areas, nbinsx=20)])
                                    fig1.update_layout(
                                        title="病灶面积分布",
                                        xaxis_title="面积 (mm²)",
                                        yaxis_title="频数"
                                    )
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with fig_col2:
                                    # 参数散点图
                                    areas = [d["physical_area_mm2"] for d in quant_info["quantitative_data"]]
                                    perimeters = [d["perimeter_mm"] for d in quant_info["quantitative_data"]]
                                    classes = [d["class_name"] for d in quant_info["quantitative_data"]]
                                    
                                    fig2 = go.Figure()
                                    unique_classes = list(set(classes))
                                    for cls in unique_classes:
                                        cls_areas = [areas[i] for i, c in enumerate(classes) if c == cls]
                                        cls_perims = [perimeters[i] for i, c in enumerate(classes) if c == cls]
                                        fig2.add_trace(go.Scatter(
                                            x=cls_areas, y=cls_perims,
                                            mode='markers',
                                            name=cls,
                                            marker=dict(size=10)
                                        ))
                                    
                                    fig2.update_layout(
                                        title="面积 vs 周长散点图",
                                        xaxis_title="面积 (mm²)",
                                        yaxis_title="周长 (mm)",
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                            
                            # 导出选项
                            st.subheader("导出分析结果")
                            exp_col1, exp_col2, exp_col3 = st.columns(3)
                            
                            with exp_col1:
                                # 导出为CSV
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 导出CSV数据",
                                    data=csv,
                                    file_name=f"quantitative_analysis_{img_name.split('.')[0]}.csv",
                                    mime="text/csv",
                                    key=f"csv_{img_name}"
                                )
                            
                            with exp_col2:
                                # 导出为JSON
                                json_str = json.dumps(report, indent=2, ensure_ascii=False)
                                st.download_button(
                                    label="📥 导出JSON报告",
                                    data=json_str.encode('utf-8'),
                                    file_name=f"quantitative_report_{img_name.split('.')[0]}.json",
                                    mime="application/json",
                                    key=f"json_{img_name}"
                                )
                            
                            with exp_col3:
                                # 导出为PDF
                                if st.session_state.current_case_id and st.session_state.current_case_id in st.session_state.case_studies:
                                    case_data = {img_name: quant_info}
                                    pdf_buffer = generate_pdf_report(case_data, f"single_{img_name.split('.')[0]}")
                                    if pdf_buffer:
                                        st.download_button(
                                            label="📥 导出PDF报告",
                                            data=pdf_buffer,
                                            file_name=f"quantitative_report_{img_name.split('.')[0]}.pdf",
                                            mime="application/pdf",
                                            key=f"pdf_{img_name}"
                                        )
                    else:
                        st.info("该图片未检测到病灶区域，无量化分析数据。")
                
                st.divider()
    
    # 标签页3: 对比分析
    with tab3:
        st.header("📈 病例对比分析")
        
        if not st.session_state.current_case_id or st.session_state.current_case_id not in st.session_state.case_studies:
            st.info("请先选择或创建一个病例，并进行多次检查以查看对比分析。")
        else:
            case_data = st.session_state.case_studies[st.session_state.current_case_id]
            
            if len(case_data) < 2:
                st.info(f"病例 **{st.session_state.current_case_id}** 目前只有 {len(case_data)} 次检查记录。至少需要2次检查才能进行对比分析。")
            else:
                st.success(f"病例 **{st.session_state.current_case_id}** 共有 {len(case_data)} 次检查记录。")
                
                # 显示病例时间线
                st.subheader("检查时间线")
                dates = sorted(case_data.keys())
                for i, date in enumerate(dates):
                    data = case_data[date]
                    region_count = len(data["quantitative_data"]) if data["quantitative_data"] else 0
                    st.write(f"{i+1}. **{date}** - 图像: {data['image_name']} - 检测区域: {region_count}个")
                
                # 生成对比图表
                st.subheader("变化趋势分析")
                comparison_chart = create_comparison_chart(case_data)
                if comparison_chart:
                    st.plotly_chart(comparison_chart, use_container_width=True)
                else:
                    st.warning("无法生成对比图表，请确保有足够的检查数据。")
                
                # 详细对比表格
                st.subheader("详细对比数据")
                comparison_data = []
                for date in dates:
                    data = case_data[date]
                    if data["quantitative_data"]:
                        areas = [d["physical_area_mm2"] for d in data["quantitative_data"]]
                        diameters = [d["max_diameter_mm"] for d in data["quantitative_data"]]
                        perimeters = [d["perimeter_mm"] for d in data["quantitative_data"]]
                        
                        comparison_data.append({
                            "检查时间": date,
                            "图像名称": data["image_name"],
                            "检测区域数": len(data["quantitative_data"]),
                            "总面积(mm²)": round(sum(areas), 2),
                            "平均面积(mm²)": round(np.mean(areas), 2) if areas else 0,
                            "最大面积(mm²)": round(max(areas), 2) if areas else 0,
                            "平均直径(mm)": round(np.mean(diameters), 2) if diameters else 0,
                            "最大直径(mm)": round(max(diameters), 2) if diameters else 0,
                            "平均周长(mm)": round(np.mean(perimeters), 2) if perimeters else 0
                        })
                    else:
                        comparison_data.append({
                            "检查时间": date,
                            "图像名称": data["image_name"],
                            "检测区域数": 0,
                            "总面积(mm²)": 0,
                            "平均面积(mm²)": 0,
                            "最大面积(mm²)": 0,
                            "平均直径(mm)": 0,
                            "最大直径(mm)": 0,
                            "平均周长(mm)": 0
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # 计算变化率
                if len(comparison_data) >= 2:
                    st.subheader("变化率分析")
                    latest = comparison_data[-1]
                    previous = comparison_data[-2]
                    
                    col_chg1, col_chg2, col_chg3 = st.columns(3)
                    
                    with col_chg1:
                        if previous["总面积(mm²)"] > 0:
                            area_change = ((latest["总面积(mm²)"] - previous["总面积(mm²)"]) / previous["总面积(mm²)"]) * 100
                            st.metric("总面积变化", f"{latest['总面积(mm²)']} mm²", 
                                    delta=f"{area_change:.1f}%", 
                                    delta_color="inverse" if area_change > 0 else "normal")
                    
                    with col_chg2:
                        if previous["平均面积(mm²)"] > 0:
                            avg_area_change = ((latest["平均面积(mm²)"] - previous["平均面积(mm²)"]) / previous["平均面积(mm²)"]) * 100
                            st.metric("平均面积变化", f"{latest['平均面积(mm²)']} mm²", 
                                    delta=f"{avg_area_change:.1f}%", 
                                    delta_color="inverse" if avg_area_change > 0 else "normal")
                    
                    with col_chg3:
                        region_change = latest["检测区域数"] - previous["检测区域数"]
                        st.metric("病灶数量变化", latest["检测区域数"], 
                                delta=f"{'+' if region_change > 0 else ''}{region_change}", 
                                delta_color="inverse" if region_change > 0 else "normal")
                
                # 导出对比报告
                st.subheader("导出对比报告")
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    # 导出对比数据为CSV
                    csv_data = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 导出对比数据 (CSV)",
                        data=csv_data,
                        file_name=f"comparison_{st.session_state.current_case_id}.csv",
                        mime="text/csv"
                    )
                
                with exp_col2:
                    # 导出完整PDF报告
                    pdf_buffer = generate_pdf_report(case_data, st.session_state.current_case_id)
                    if pdf_buffer:
                        st.download_button(
                            label="📥 导出完整报告 (PDF)",
                            data=pdf_buffer,
                            file_name=f"medical_report_{st.session_state.current_case_id}.pdf",
                            mime="application/pdf"
                        )

# ------------------ 应用主逻辑 ------------------
if not st.session_state.authenticated:
    login_page()
else:
    # 在侧边栏显示登出按钮
    if st.sidebar.button("🚪 退出登录"):
        st.session_state.authenticated = False
        st.session_state.model = None
        st.session_state.uploaded_images = []
        st.session_state.processed_results = {}
        st.session_state.quantitative_results = {}
        st.session_state.case_studies = {}
        st.session_state.current_case_id = None
        st.rerun()
    
    # 显示主应用界面
    main_app()
    
    # 页脚信息
    st.sidebar.divider()
    st.sidebar.caption("""
    **使用说明:**
    1. 通过侧边栏加载或上传训练好的YOLO模型
    2. 创建或选择病例，输入病例ID
    3. 选择单张、多张图片或ZIP文件夹上传
    4. 调整检测参数和像素尺寸
    5. 点击"开始分割处理"按钮
    6. 在标签页中查看分割结果、量化分析和对比分析
    7. 导出分析报告和对比数据
    """)