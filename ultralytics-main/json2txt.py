import json
import os

# 读取Labelme的JSON文件
def convert_labelme_to_yolov8(json_file, output_txt, class_mapping):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 提取图像的宽度和高度
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    with open(output_txt, 'w') as f:
        for shape in data['shapes']:
            # 获取标注类别和点的坐标
            label = shape['label']
            points = shape['points']

            # 根据类别映射表转换为class_id
            class_id = class_mapping.get(label, None)
            if class_id is None:
                continue  # 跳过没有定义的类别

            # 将坐标归一化
            normalized_points = []
            for point in points:
                x, y = point
                x_norm = x / image_width
                y_norm = y / image_height
                normalized_points.extend([x_norm, y_norm])

            # 输出 YOLOv8 格式： <class_id> <x1> <y1> ... <xn> <yn>
            f.write(f"{class_id} " + " ".join(map(str, normalized_points)) + "\n")

# 使用类别映射表，假设你有三个类别：cat、dog、person
class_mapping = {
    "remote" : 0,
    "toy" : 1,
    "game" : 2,
    "clock" : 3,
    "pen" : 4,
    "ball" : 5
}

# 调用函数进行转换
json_file = "10.json"
output_txt = "10.txt"
convert_labelme_to_yolov8(json_file, output_txt, class_mapping)
