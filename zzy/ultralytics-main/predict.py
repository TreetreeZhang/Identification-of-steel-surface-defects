from ultralytics import YOLO

# 加载训练好的模型，替换成你的模型文件路径
model = YOLO('runs\\segment\\train5\\weights\\best.pt')

# 指定图片所在文件夹路径
image_folder = 'test/'

# 进行批量预测
results = model.predict(source=image_folder, save=True)