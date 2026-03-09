from ultralytics import YOLO
import os
import torch
from datetime import datetime

def main():
    """主训练函数"""
    # 配置训练参数
    config = {
        "model": "yolov8n-seg.pt",  # 分割模型
        "data_yaml": "data.yaml",   # 数据配置文件
        "epochs": 100,              # 训练轮数
        "imgsz": 512,               # 图片大小
        "batch": 8,                 # 批次大小
        "device": "0",              # 设备: 0表示第一个GPU
    }
    
    print("=" * 60)
    print("YOLOv8 分割模型训练")
    print("=" * 60)
    
    # 1. 检查数据配置文件
    if not os.path.exists(config["data_yaml"]):
        print(f"❌ 错误: 找不到{config['data_yaml']}文件")
        print("请确保data.yaml在当前目录")
        return
    
    # 2. 检查设备
    if config["device"].isdigit():  # 如果是数字，表示GPU编号
        device_id = int(config["device"])
        if torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                print(f"✅ 使用GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            else:
                print(f"⚠️ GPU {device_id} 不可用，将使用CPU")
                config["device"] = "cpu"
        else:
            print("⚠️ GPU不可用，将使用CPU")
            config["device"] = "cpu"
    
    # 3. 生成时间戳用于日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"train_{timestamp}"
    
    # 4. 打印训练配置
    print(f"模型: {config['model']}")
    print(f"数据: {config['data_yaml']}")
    print(f"轮数: {config['epochs']}")
    print(f"图片大小: {config['imgsz']}")
    print(f"批次: {config['batch']}")
    print(f"设备: {config['device']}")
    print("-" * 60)
    print("TensorBoard: 启用")
    print(f"实验名称: {exp_name}")
    print("TensorBoard日志将自动生成")
    print("训练完成后，使用以下命令启动TensorBoard：")
    print("  tensorboard --logdir runs")
    print("=" * 60)
    
    # 5. 加载模型
    print("加载模型...")
    model = YOLO(config["model"])
    
    # 6. 开始训练
    print("\n开始训练...")
    print("=" * 40)
    
    try:
        results = model.train(
            data=config["data_yaml"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            device=config["device"],
            patience=20,      # 早停耐心值
            save=True,        # 保存检查点
            save_period=10,   # 每10轮保存一次
            workers=4,        # 数据加载线程数
            exist_ok=True,    # 允许覆盖
            
            # TensorBoard相关参数
            project="runs",   # 项目目录
            name="train",    # 使用时间戳作为运行名称
            plots=True      # 生成训练图
        )
        
        print("✅ 训练完成!")
        
        # 7. 验证模型性能
        print("\n验证模型性能...")
        metrics = model.val()
        
        if hasattr(metrics, "box") and metrics.box is not None:
            print(f"边界框指标:")
            print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
            print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
        
        if hasattr(metrics, "seg") and metrics.seg is not None:
            print(f"分割指标:")
            print(f"  - 掩码mAP@0.5: {metrics.seg.map50:.4f}")
            print(f"  - 掩码mAP@0.5:0.95: {metrics.seg.map:.4f}")
        
        # 8. 显示最佳模型路径
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'best'):
            best_model = model.trainer.best
            print(f"\n最佳模型: {best_model}")
        
        # 9. 自动查找TensorBoard日志目录
        print("=" * 60)
        print("训练流程完成!")
        print("\nTensorBoard使用指南:")
        print("1. 在终端中运行以下命令启动TensorBoard:")
        print("   tensorboard --logdir runs")
        print("2. 然后在浏览器中访问: http://localhost:6006")
        print("3. 如果6006端口被占用，可以指定其他端口:")
        print("   tensorboard --logdir runs --port 6007")
        
        # 自动检测并显示最新的训练目录
        runs_dir = "runs"
        if os.path.exists(runs_dir):
            subdirs = [d for d in os.listdir(runs_dir) 
                      if os.path.isdir(os.path.join(runs_dir, d))]
            
            if subdirs:
                # 按修改时间排序
                sorted_dirs = sorted(
                    [os.path.join(runs_dir, d) for d in subdirs],
                    key=lambda x: os.path.getmtime(x),
                    reverse=True
                )
                print(f"\n检测到的训练目录:")
                for i, dir_path in enumerate(sorted_dirs[:3]):  # 显示最新的3个
                    print(f"  {i+1}. {dir_path}")
                print(f"\nTensorBoard将自动加载所有训练日志")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)

if __name__ == "__main__":
    main()