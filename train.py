from ultralytics import YOLO

if __name__ == "__main__":
    # 实例分割任务
    model = YOLO("yolo26n-seg.pt")

    # 开始训练
    train = model.train(
        data="data.yaml",
        epochs=300,           # SGD至少要300轮以上
        imgsz=640,
        batch=-1,             # 自动60%的GPU使用率
        device="0",
        patience=20,
        visualize=True,
        verbose=True,
        optimizer='SGD',      # AdamW是错的，过拟合严重
        lr0=1e-3,             # SGD是对的，大道至简
        momentum=0.937,
        weight_decay=0.0005,  # SGD尽可能需要小的预热
        warmup_epochs=3.0,    # 同上
        warmup_momentum=0.8,  # 同上
        warmup_bias_lr=0.1,   # 同上
        box=7.5,              # v5开始的bbox损失
        cls=0.5,              # 同上
        dfl=1.5,              # v26的损失函数移除了dfl,并去掉了单独nms的流程
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        profile=True,         # onnx + tensorRT
        compile=True,         # python-triton，编译期优化
        save=True,
        save_period=10,
        workers=4,
        exist_ok=True,
        plots=True
    )