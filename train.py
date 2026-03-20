from ultralytics import YOLO

if __name__ == "__main__":
    # 实例分割任务
    model = YOLO("yolo26n-seg.pt")

    # 开始训练
    train = model.train(
        data="data.yaml",
        epochs=300,           # 100太少了
        imgsz=640,
        batch=-1,
        device="0",
        patience=20,
        visualize=True,
        verbose=True,
        optimizer='SGD',      # adamw是错的
        lr0=1e-3,             # sgd是对的
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        profile=True,         # onnx + tensorRT
        compile=True,         # python-triton，编译期优化确实快
        save=True,
        save_period=10,
        workers=4,
        exist_ok=True,
        plots=True
    )