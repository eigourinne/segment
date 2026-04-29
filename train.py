from ultralytics import YOLO

if __name__ == "__main__":
    # 医学实例分割任务，使用ELA改进
    model = YOLO("yolo26n-seg-meta.yaml").load("yolo26n-seg.pt")

    # 开始训练
    train = model.train(
        data="data.yaml",
        task="segment",
        epochs=300,           # SGD至少要300轮以上
        imgsz=640,
        batch=0.9,             # 限制GPU使用率
        device="0",
        patience=20,
        visualize=True,
        verbose=True,
        optimize=True,
        optimizer='SGD',      # SGD更稳定
        lr0=1e-3,             # 同上
        momentum=0.937,       # 同上
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,    # 同上
        warmup_momentum=0.8,  # 同上
        warmup_bias_lr=0.1,   # 同上
        cos_lr=True,          # 余弦退火
        box=7.5,              # v5开始的bbox损失
        cls=0.5,              # 同上
        dfl=0,                # v26的损失函数可以移除dfl
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.2,          # 防过拟合
        val=True,
        profile=True,         # onnx + tensorRT
        compile=True,         # python-triton，编译期优化
        save=True,
        save_period=10,
        workers=4,
        exist_ok=True,
        plots=True
    )