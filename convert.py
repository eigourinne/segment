from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
# 导出一个动态batch的ONNX，方便查看结构
success = model.export(format="onnx", dynamic=True, simplify=True)
# 导出的文件默认与权重文件同名，后缀为 .onnx