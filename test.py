import cv2
import os
import glob
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体，解决乱码问题
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import queue

class WoundSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("医学图像分割检测系统")
        self.root.geometry("1000x700")
        
        # 模型变量
        self.model = None
        self.model_path = ""
        
        # 图片路径列表
        self.image_paths = []
        self.current_image_index = 0
        
        # 创建GUI组件
        self.create_widgets()
        
        # 自动加载模型
        self.load_model_auto()
    
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="YOLOv26 医学分割检测系统", 
                                font=("Microsoft YaHei", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 模型状态显示
        self.model_status_label = ttk.Label(main_frame, text="模型状态: 未加载", 
                                           font=("Microsoft YaHei", 10))
        self.model_status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # 选择图片按钮
        ttk.Button(main_frame, text="选择图片", 
                  command=self.select_image, width=15).grid(row=2, column=0, pady=5, padx=5)
        
        # 选择文件夹按钮
        ttk.Button(main_frame, text="选择文件夹", 
                  command=self.select_folder, width=15).grid(row=2, column=1, pady=5, padx=5)
        
        # 手动输入模型路径按钮
        ttk.Button(main_frame, text="手动选择模型", 
                  command=self.select_model, width=15).grid(row=2, column=2, pady=5, padx=5)
        
        # 路径显示区域
        ttk.Label(main_frame, text="当前选择的图片:", 
                 font=("Microsoft YaHei", 10)).grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        
        # 列表框显示图片路径
        self.listbox_frame = ttk.Frame(main_frame)
        self.listbox_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.path_listbox = tk.Listbox(self.listbox_frame, height=8, width=80,
                                      yscrollcommand=scrollbar.set, font=("Consolas", 9))
        self.path_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.path_listbox.yview)
        
        # 清空列表按钮
        ttk.Button(main_frame, text="清空列表", 
                  command=self.clear_list, width=15).grid(row=5, column=0, pady=5)
        
        # 处理按钮
        self.process_button = ttk.Button(main_frame, text="开始处理", 
                                        command=self.process_images, width=20, state=tk.DISABLED)
        self.process_button.grid(row=5, column=1, pady=5)
        
        # 进度标签
        self.progress_label = ttk.Label(main_frame, text="就绪", font=("Microsoft YaHei", 10))
        self.progress_label.grid(row=6, column=0, columnspan=3, pady=(10, 5))
        
        # 进度条
        self.progress_bar = ttk.Progressbar(main_frame, length=400, mode='indeterminate')
        self.progress_bar.grid(row=7, column=0, columnspan=3, pady=(0, 20))
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="", font=("Microsoft YaHei", 9))
        self.status_label.grid(row=8, column=0, columnspan=3, pady=(0, 10))
        
        # 图片预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="图片预览", padding="10")
        preview_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # 配置预览框架网格
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # 原图预览
        ttk.Label(preview_frame, text="原图", font=("Microsoft YaHei", 10)).grid(row=0, column=0, pady=(0, 5))
        self.original_preview_label = ttk.Label(preview_frame, text="暂无图片", 
                                               relief=tk.SUNKEN, width=40, anchor=tk.CENTER)
        self.original_preview_label.grid(row=1, column=0, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果预览
        ttk.Label(preview_frame, text="分割结果", font=("Microsoft YaHei", 10)).grid(row=0, column=1, pady=(0, 5))
        self.result_preview_label = ttk.Label(preview_frame, text="等待处理", 
                                             relief=tk.SUNKEN, width=40, anchor=tk.CENTER)
        self.result_preview_label.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def load_model_auto(self):
        """自动查找并加载模型"""
        possible_paths = [
            "runs/segment/train/weights/best.pt",
            "./best.pt"
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                self.model_path = model_path
                self.model_status_label.config(text=f"模型状态: 已加载 ({model_path})")
                try:
                    self.model = YOLO(model_path)
                    messagebox.showinfo("成功", f"模型加载成功: {model_path}")
                    return
                except Exception as e:
                    messagebox.showerror("错误", f"模型加载失败: {e}")
                    return
        
        # 如果自动查找失败，提示用户手动选择
        self.model_status_label.config(text="模型状态: 未找到模型文件")
        response = messagebox.askyesno("模型未找到", "未找到模型文件 'best.pt'，是否手动选择?")
        if response:
            self.select_model()
    
    def select_model(self):
        """手动选择模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型文件", "*.pt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.model_path = file_path
            try:
                self.model = YOLO(file_path)
                self.model_status_label.config(text=f"模型状态: 已加载 ({os.path.basename(file_path)})")
                messagebox.showinfo("成功", f"模型加载成功: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {e}")
    
    def select_image(self):
        """选择单张图片"""
        file_paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_paths:
            for file_path in file_paths:
                if file_path not in self.image_paths:
                    self.image_paths.append(file_path)
                    self.path_listbox.insert(tk.END, file_path)
            
            self.update_process_button_state()
    
    def select_folder(self):
        """选择文件夹"""
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")
        
        if folder_path:
            # 递归查找文件夹下所有支持的图片文件
            image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
            new_images = []
            
            for ext in image_extensions:
                found_images = glob.glob(os.path.join(folder_path, '**', ext), recursive=True)
                for img_path in found_images:
                    if img_path not in self.image_paths:
                        new_images.append(img_path)
            
            if new_images:
                for img_path in new_images:
                    self.image_paths.append(img_path)
                    self.path_listbox.insert(tk.END, img_path)
                
                self.update_process_button_state()
                messagebox.showinfo("成功", f"找到 {len(new_images)} 张新图片")
            else:
                messagebox.showwarning("警告", f"在文件夹中未找到任何支持的图片文件")
    
    def clear_list(self):
        """清空图片列表"""
        self.image_paths.clear()
        self.path_listbox.delete(0, tk.END)
        self.update_process_button_state()
    
    def update_process_button_state(self):
        """更新处理按钮状态"""
        if self.model is not None and len(self.image_paths) > 0:
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)
    
    def update_preview(self, original_img_path, result_img_path=None):
        """更新预览图片"""
        # 更新原图预览
        try:
            img = Image.open(original_img_path)
            img.thumbnail((300, 300))  # 缩放到合适大小
            photo = ImageTk.PhotoImage(img)
            self.original_preview_label.config(image=photo, text="")
            self.original_preview_label.image = photo  # 保持引用
        except Exception as e:
            self.original_preview_label.config(image=None, text=f"预览失败: {e}")
        
        # 更新结果预览
        if result_img_path and os.path.exists(result_img_path):
            try:
                img = Image.open(result_img_path)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.result_preview_label.config(image=photo, text="")
                self.result_preview_label.image = photo
            except Exception as e:
                self.result_preview_label.config(image=None, text=f"结果预览失败")
        else:
            self.result_preview_label.config(image=None, text="等待处理")
    
    def process_images(self):
        """处理图片"""
        if not self.model:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        if not self.image_paths:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        # 禁用按钮，开始处理
        self.process_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.progress_label.config(text="正在处理中...")
        
        # 在新线程中处理图片，避免GUI卡顿
        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()
    
    def process_images_thread(self):
        """处理图片的线程函数"""
        try:
            total_images = len(self.image_paths)
            
            for i, img_path in enumerate(self.image_paths, 1):
                # 更新状态
                self.root.after(0, self.update_progress, i, total_images, img_path)
                
                try:
                    # 处理图片
                    original_img, result_img, has_detection = self.predict_and_plot(img_path)
                    
                    # 保存结果图片
                    result_dir = "results"
                    os.makedirs(result_dir, exist_ok=True)
                    result_filename = f"result_{os.path.basename(img_path)}"
                    result_path = os.path.join(result_dir, result_filename)
                    
                    # 保存结果图片
                    plt.figure(figsize=(12, 6))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(original_img)
                    plt.title(f'原图: {os.path.basename(img_path)}', fontproperties='Microsoft YaHei')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(result_img)
                    detection_status = "(检测到医学)" if has_detection else "(未检测到医学)"
                    plt.title(f'分割结果 {detection_status}', fontproperties='Microsoft YaHei')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(result_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # 更新GUI预览
                    self.root.after(0, self.update_preview, img_path, result_path)
                    
                    # 显示处理完成的图片
                    self.root.after(0, self.show_result, result_path, has_detection)
                    
                except Exception as e:
                    error_msg = f"处理图片失败: {os.path.basename(img_path)}\n错误: {str(e)}"
                    self.root.after(0, messagebox.showerror, "处理失败", error_msg)
            
            # 处理完成
            self.root.after(0, self.processing_complete, total_images)
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "错误", f"处理过程中发生错误: {str(e)}")
        finally:
            self.root.after(0, self.reset_ui)
    
    def predict_and_plot(self, img_path, conf_threshold=0.25):
        """对单张图片进行预测"""
        # 读取原图
        original_img = cv2.imread(img_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 使用模型进行预测
        results = self.model(img_path, conf=conf_threshold)
        
        # 创建一个用于绘制预测结果的副本
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            plotted_img = result.plot(masks=True, boxes=False, labels=False)
            plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        else:
            plotted_img_rgb = original_img_rgb.copy()
        
        return original_img_rgb, plotted_img_rgb, len(results) > 0 and results[0].masks is not None
    
    def update_progress(self, current, total, img_path):
        """更新进度显示"""
        self.progress_label.config(text=f"正在处理: {current}/{total} - {os.path.basename(img_path)}")
        self.status_label.config(text=f"当前处理: {os.path.basename(img_path)}")
    
    def show_result(self, result_path, has_detection):
        """显示处理结果"""
        # 在新窗口中显示结果
        result_window = tk.Toplevel(self.root)
        result_window.title("分割结果")
        result_window.geometry("1200x600")
        
        # 加载图片
        img = Image.open(result_path)
        img.thumbnail((1000, 500))
        photo = ImageTk.PhotoImage(img)
        
        # 显示图片
        label = ttk.Label(result_window, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
        
        # 显示检测状态
        status = "检测到医学" if has_detection else "未检测到医学"
        status_label = ttk.Label(result_window, text=f"状态: {status}", 
                                font=("Microsoft YaHei", 12))
        status_label.pack(pady=(0, 10))
        
        # 保存路径显示
        path_label = ttk.Label(result_window, text=f"保存路径: {result_path}", 
                              font=("Microsoft YaHei", 9))
        path_label.pack(pady=(0, 10))
        
        # 关闭按钮
        ttk.Button(result_window, text="关闭", 
                  command=result_window.destroy).pack(pady=(0, 10))
    
    def processing_complete(self, total_images):
        """处理完成"""
        messagebox.showinfo("完成", f"所有图片处理完成！共处理 {total_images} 张图片。\n结果保存在 'results' 文件夹中。")
    
    def reset_ui(self):
        """重置UI状态"""
        self.progress_bar.stop()
        self.progress_label.config(text="处理完成")
        self.status_label.config(text="")
        self.process_button.config(state=tk.NORMAL)
        self.result_preview_label.config(image=None, text="处理完成")

def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置窗口图标（如果有的话）
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = WoundSegmentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()