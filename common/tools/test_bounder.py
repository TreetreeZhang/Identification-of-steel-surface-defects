# Description: 将符合yolo标签的txt文件中的坐标转换为图像上的多边形边界框，并绘制在图像上

import cv2
import numpy as np

# 读取图像
image_path = '000201.jpg'  # 请替换为你的图像文件路径
image = cv2.imread(image_path)

# 图像的实际尺寸
image_height, image_width = 200, 200  # 这里假设你的图像是200x200像素

# 读取txt文件中的坐标
txt_path = '000201.txt'  # 请替换为你的txt文件路径
with open(txt_path, 'r') as f:
    data = f.readlines()

# 处理txt文件的每一行
for line in data:
    parts = line.strip().split()
    
    # 提取类别（忽略，因为所有类别为1）以及后面的坐标
    coords = parts[1:]
    
    # 每对数值是x, y对，先保存到一个列表
    polygon_points = []
    for i in range(0, len(coords), 2):
        x_relative = float(coords[i])
        y_relative = float(coords[i + 1])
        
        # 将相对坐标转换为像素坐标
        x_pixel = int(x_relative * image_width)
        y_pixel = int(y_relative * image_height)
        
        # 保存点坐标
        polygon_points.append((x_pixel, y_pixel))

    # 将点转换为numpy数组并绘制多边形
    polygon_points = np.array(polygon_points, np.int32)
    polygon_points = polygon_points.reshape((-1, 1, 2))

    # 画出边界 (红色边框，粗细为2)
    cv2.polylines(image, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)

# 保存结果
output_path = 'output_image.jpg'  # 保存结果的路径
cv2.imwrite(output_path, image)

# 显示图像 (可选)
cv2.imshow('Image with Boundaries', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
