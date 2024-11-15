import os

def count_images_in_folder(folder_path):
    # 常见图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_count = 0

    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 获取文件扩展名并转换为小写
            ext = os.path.splitext(file)[-1].lower()
            # 如果扩展名在图片扩展名集合中，则计数加一
            if ext in image_extensions:
                image_count += 1

    return image_count

# 使用示例
folder_path = '/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train/label'
print(f"文件夹 '{folder_path}' 中的图片文件总数为: {count_images_in_folder(folder_path)}")
