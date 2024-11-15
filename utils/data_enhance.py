import os
import random
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

def crop_images_from_single_image(image, mask, num_parts):
    # 获取原始图像的宽和高
    width, height = image.size

    cropped_image_parts = []
    cropped_mask_parts = []

    if num_parts == 4:
        # 对于4个部分，裁剪100x100的区域，起始位置随机
        crop_width, crop_height = 100, 100
        for _ in range(num_parts):
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            cropped_image_parts.append(image.crop((left, top, left + crop_width, top + crop_height)))
            cropped_mask_parts.append(mask.crop((left, top, left + crop_width, top + crop_height)))
    elif num_parts == 3:
        # 对于3个部分，保持原有逻辑
        crop_width = width // num_parts
        for i in range(num_parts):
            left = i * crop_width
            right = (i + 1) * crop_width if i != num_parts - 1 else width
            cropped_image_parts.append(image.crop((left, 0, right, height)))
            cropped_mask_parts.append(mask.crop((left, 0, right, height)))
    elif num_parts == 2:
        # 对于2个部分，裁剪宽度可以随机，起始位置随机
        random_width = random.randint(width // 4, width // 2)
        left_1 = random.randint(0, width - random_width)
        left_2 = random.randint(0, width - random_width)
        cropped_image_parts.append(image.crop((left_1, 0, left_1 + random_width, height)))
        cropped_mask_parts.append(mask.crop((left_1, 0, left_1 + random_width, height)))
        cropped_image_parts.append(image.crop((left_2, 0, left_2 + (width - random_width), height)))
        cropped_mask_parts.append(mask.crop((left_2, 0, left_2 + (width - random_width), height)))

    stitched_image = blend_images(cropped_image_parts, width, height)
    stitched_mask = blend_images(cropped_mask_parts, width, height)
    return stitched_image, stitched_mask

def blend_images(parts, width, height):
    # 使用渐入渐出法对拼接区域进行平滑过渡
    result = np.zeros((height, width, 3), dtype=np.float32)
    num_parts = len(parts)
    overlap_width = 20  # 重叠区域的宽度，用于渐入渐出

    x_offset = 0
    for i, part in enumerate(parts):
        part_np = np.array(part).astype(np.float32)
        part_width = part_np.shape[1]

        if i > 0:
            blend_width = min(overlap_width, part_width, result.shape[1] - x_offset)
            alpha = np.linspace(0, 1, blend_width).reshape(1, -1, 1)

            result[:, x_offset:x_offset + blend_width] = (
                result[:, x_offset:x_offset + blend_width] * (1 - alpha) +
                part_np[:, :blend_width] * alpha
            )
            x_offset += blend_width

        result[:, x_offset:x_offset + part_width - overlap_width] = part_np[:, overlap_width:]
        x_offset += part_width - overlap_width

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)

def process_folder(image_folder, label_folder, output_image_folder, output_label_folder):
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    labels = sorted([f for f in os.listdir(label_folder) if f.endswith('.png')])

    assert len(images) == len(labels), "图像文件和标注文件数量不匹配！"

    ensure_dir(output_image_folder)
    ensure_dir(output_label_folder)

    idx = 0
    for img_file, label_file in tqdm(zip(images, labels), total=len(images), desc="Processing files"):
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, label_file)

        img = Image.open(img_path)
        label = Image.open(label_path)

        num_parts = random.choice([2, 3, 4])
        stitched_image, stitched_label = crop_images_from_single_image(img, label, num_parts)

        stitched_image.save(os.path.join(output_image_folder, f'stitched_{idx}_{num_parts}_images.png'))
        stitched_label.save(os.path.join(output_label_folder, f'stitched_{idx}_{num_parts}_labels.png'))

        idx += 1

def process_dataset(base_path, folder_option, output_base_path=None):
    if output_base_path is None:
        output_base_path = base_path  # 如果没有指定输出路径，则使用原始路径

    if folder_option == 'train':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label')
        )
    elif folder_option == 'test':
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label')
        )
    elif folder_option == 'all':
        process_folder(
            os.path.join(base_path, 'train/image'),
            os.path.join(base_path, 'train/label'),
            os.path.join(output_base_path, 'train/image'),
            os.path.join(output_base_path, 'train/label')
        )
        process_folder(
            os.path.join(base_path, 'test/image'),
            os.path.join(base_path, 'test/label'),
            os.path.join(output_base_path, 'test/image'),
            os.path.join(output_base_path, 'test/label')
        )
    else:
        print("Invalid folder option! Please use 'train', 'test', or 'all'.")

if __name__ == "__main__":
    dataset_option = 'train'  # 替换为您想要的选项：'train', 'test', 或 'all'
    base_path = "/root/autodl-tmp/standard_project/datasets/datasets_A"  # 数据集的基本路径
    output_path = "/root/autodl-tmp/standard_project/datasets/datasets_A_splice"  # 替换为您想要的输出路径
    ensure_dir(output_path)

    process_dataset(base_path, dataset_option, output_path)



