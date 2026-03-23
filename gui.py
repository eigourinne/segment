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

# ------------------ 页面配置 ------------------
st.set_page_config(
    page_title="YOLOv26 医学图像分割系统",
    page_icon="🩹",
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
if 'conf_threshold' not in st.session_state:
    st.session_state.conf_threshold = 0.25

# ------------------ 辅助函数 ------------------
def auto_load_model():
    """自动查找并加载模型，参考自 test.py 的 load_model_auto 函数"""
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

def predict_and_plot(img_path, conf_threshold=0.25):
    """对单张图片进行预测并返回可视化结果，逻辑参考自 test.py 的 predict_and_plot 方法"""
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
    
    if len(results) > 0 and results[0].masks is not None:
        result = results[0]
        plotted_img = result.plot(masks=True, boxes=True, labels=True)  # 显示掩码、框和标签
        result_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        has_detection = True
        
        # 提取检测信息
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                detections.append({
                    "id": i + 1,
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": result.names[cls_id] if hasattr(result, 'names') else f"Class {cls_id}"
                })
    else:
        result_img_rgb = original_img_rgb.copy()
    
    return original_img_rgb, result_img_rgb, has_detection, detections

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
            time.sleep(1)  # 短暂延迟，让用户看到成功信息
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
        # 保存上传的模型到临时文件并加载
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
    
    # 3. 图片上传
    st.sidebar.header("3. 图片上传")
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
            # 重置并保存图片
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
            import zipfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(tmp_dir)
                # 递归查找所有图片文件
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(Path(tmp_dir).rglob(f"*{ext}"))
                
                if image_paths:
                    # 转换为类文件对象以便统一处理
                    from io import BytesIO
                    uploaded_images = []
                    for img_path in image_paths:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                        # 创建一个类文件对象，需要文件名和数据
                        file_like_object = BytesIO(img_bytes)
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
        st.sidebar.success("已清空图片列表。")
    
    # 4. 处理按钮
    st.sidebar.header("4. 执行处理")
    process_button = st.sidebar.button(
        "🚀 开始分割处理",
        type="primary",
        disabled=(st.session_state.model is None or len(st.session_state.uploaded_images) == 0),
        help="加载模型并选择图片后即可使用。"
    )
    
    # 主界面
    st.title("🩹 YOLOv26 医学分割可视化系统")
    
    # 状态显示
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        model_status = "✅ 已加载" if st.session_state.model is not None else "❌ 未加载"
        st.metric("模型状态", model_status, delta=st.session_state.model_path if st.session_state.model_path else "N/A")
    with col_status2:
        st.metric("待处理图片", len(st.session_state.uploaded_images))
    
    st.divider()
    
    # 图片预览与处理区域
    if len(st.session_state.uploaded_images) > 0:
        st.subheader(f"📸 已选择图片 ({len(st.session_state.uploaded_images)} 张)")
        
        # 显示图片缩略图网格
        cols = st.columns(4)
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            with cols[idx % 4]:
                # 转换为PIL Image显示
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, width=400)
                st.caption(f"{idx+1}. {img_file.name}")
        
        # 当点击处理按钮时
        if process_button:
            st.subheader("🔄 处理进度与结果")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 临时保存上传的图片并进行处理
            for idx, img_file in enumerate(st.session_state.uploaded_images):
                status_text.text(f"正在处理: {img_file.name} ({idx+1}/{len(st.session_state.uploaded_images)})")
                progress_bar.progress((idx) / len(st.session_state.uploaded_images))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(img_file.name).suffix) as tmp_file:
                    tmp_file.write(img_file.getbuffer())
                    tmp_img_path = tmp_file.name
                
                try:
                    # 进行预测
                    original_img, result_img, has_detection, detections = predict_and_plot(
                        tmp_img_path, 
                        conf_threshold=st.session_state.conf_threshold
                    )
                    
                    # 保存结果到会话状态
                    st.session_state.processed_results[img_file.name] = {
                        "original": original_img,
                        "result": result_img,
                        "has_detection": has_detection,
                        "detections": detections,
                        "tmp_path": tmp_img_path
                    }
                    
                except Exception as e:
                    st.error(f"处理图片 `{img_file.name}` 时出错: {e}")
                finally:
                    # 清理临时文件（可选，如果不再需要原文件）
                    # os.unlink(tmp_img_path)
                    pass
            
            progress_bar.progress(1.0)
            status_text.text("✅ 所有图片处理完成！")
            st.balloons()
    
    # 显示处理结果
    if st.session_state.processed_results:
        st.divider()
        st.subheader("📊 分割结果对比")
        
        # 为每张处理过的图片创建一个展示区域
        for img_name, result_info in st.session_state.processed_results.items():
            with st.expander(f"查看详情: **{img_name}**", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**原图**")
                    st.image(result_info["original"], width=400, clamp=True)
                
                with col2:
                    detection_status = "✅ 检测到医学" if result_info["has_detection"] else "⚠️ 未检测到医学"
                    st.markdown(f"**分割结果** ({detection_status})")
                    st.image(result_info["result"], width=400, clamp=True)
                
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
                    # 将结果图片转换为字节流供下载
                    result_pil = Image.fromarray(result_info["result"])
                    from io import BytesIO
                    buf = BytesIO()
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
                    # 保存处理前后的对比图
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].imshow(result_info["original"])
                    axes[0].set_title("Original Image")
                    axes[0].axis('off')
                    
                    axes[1].imshow(result_info["result"])
                    status = "Detected" if result_info["has_detection"] else "Not Detected"
                    axes[1].set_title(f"Segmentation Result ({status})")
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    
                    # 保存对比图到缓冲区
                    buf_comparison = BytesIO()
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
        st.rerun()
    
    # 显示主应用界面
    main_app()
    
    # 页脚信息
    st.sidebar.divider()
    st.sidebar.caption("""
    **使用说明:**
    1. 通过侧边栏加载或上传训练好的YOLO模型。
    2. 选择单张、多张图片或ZIP文件夹上传。
    3. 调整置信度阈值。
    4. 点击“开始分割处理”按钮。
    5. 在下方查看结果并下载图片。
    """)