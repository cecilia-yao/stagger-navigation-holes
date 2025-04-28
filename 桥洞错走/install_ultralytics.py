from ultralytics import YOLO
model = YOLO("pillar_model.pt")
results = model("C:\\Users\\h-p\\myPC\\Desktop\\dxy\\大学\\项目组\\1.jpg", show=True)
