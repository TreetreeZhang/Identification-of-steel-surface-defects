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

    global cropped_image_four_parts, cropped_mask_four_parts
    global cropped_image_three_parts, cropped_mask_three_parts
    global cropped_image_two_parts, cropped_mask_two_parts

    global image_four_parts_buffer, mask_four_parts_buffer
    global image_three_parts_buffer, mask_three_parts_buffer
    global image_two_parts_buffer, mask_two_parts_buffer

    if num_parts == 4:
        # 对于4个部分，裁剪100x100的区域，起始位置随机
        crop_width, crop_height = 100, 100
        image_four_parts_buffer.append(image)
        mask_four_parts_buffer.append(mask)
        if len(image_four_parts_buffer) == 4:
            for i in range(4):
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                cropped_image_four_parts.append(image_four_parts_buffer[i].crop((left, top, left + crop_width, top + crop_height)))
                cropped_mask_four_parts.append(mask_four_parts_buffer[i].crop((left, top, left + crop_width, top + crop_height)))
            image_four_parts_buffer = []
            mask_four_parts_buffer = []
    elif num_parts == 3:
        # 对于3个部分，裁剪宽度保持一致
        crop_width = width // num_parts
        image_three_parts_buffer.append(image)
        mask_three_parts_buffer.append(mask)
        if len(image_three_parts_buffer) == 3:
            for i in range(num_parts):
                left = random.randint(0, width - crop_width - 2)
                right = left + crop_width
                if(i == 2):
                    right += 2
                cropped_image_three_parts.append(image_three_parts_buffer[i].crop((left, 0, right, height)))
                cropped_mask_three_parts.append(mask_three_parts_buffer[i].crop((left, 0, right, height)))
            image_three_parts_buffer = []
            mask_three_parts_buffer = []
    elif num_parts == 2:
        # 对于2个部分，随机裁剪宽度
        crop_width1 = random.randint(50, 150)
        image_two_parts_buffer.append(image)
        mask_two_parts_buffer.append(mask)
        if len(image_two_parts_buffer) == 2:
            for i in range(num_parts):
                left = random.randint(0, width - crop_width1) if i == 0 else random.randint(0, width - (width - crop_width1))
                right = left + crop_width1 if i == 0 else left + width - crop_width1
                cropped_image_two_parts.append(image_two_parts_buffer[i].crop((left, 0, right, height)))
                cropped_mask_two_parts.append(mask_two_parts_buffer[i].crop((left, 0, right, height)))
            image_two_parts_buffer = []
            mask_two_parts_buffer = []
    
    if(len(cropped_image_four_parts) == 4):
        stitched_image = blend_images(cropped_image_four_parts,4)
        stitched_mask = blend_images(cropped_mask_four_parts,4, is_mask=True)
        cropped_image_four_parts = []
        cropped_mask_four_parts = []
        return stitched_image, stitched_mask
    
    elif(len(cropped_image_three_parts) == 3):
        stitched_image = blend_images(cropped_image_three_parts, 3)
        stitched_mask = blend_images(cropped_mask_three_parts,3, is_mask=True)
        cropped_image_three_parts = []
        cropped_mask_three_parts = []
        return stitched_image, stitched_mask
    
    elif(len(cropped_image_two_parts) == 2):
        stitched_image = blend_images(cropped_image_two_parts,2)
        stitched_mask = blend_images(cropped_mask_two_parts, 2, is_mask=True)
        cropped_image_two_parts = []
        cropped_mask_two_parts = []
        return stitched_image, stitched_mask

    return None, None

from PIL import ImageChops

def blend_images(parts, part_num, is_mask=False):
    assert len(parts) == part_num, "部分数量不匹配！"

    # 设置重叠宽度
    overlap_width = 2
    
    if part_num == 4:
        # 对于4个部分的拼接
        stitched_image = Image.new('RGB', (200, 200)) if not is_mask else Image.new('L', (200, 200))
        stitched_image.paste(parts[0], (0, 0))
        stitched_image.paste(parts[1], (100, 0))
        stitched_image.paste(parts[2], (0, 100))
        stitched_image.paste(parts[3], (100, 100))
        
    elif part_num == 3:
        # 对于3个部分的拼接
        width = sum([part.width for part in parts])  # 总宽度为三部分宽度之和
        height = parts[0].height  # 假设高度相同
        stitched_image = Image.new('RGB', (width, height)) if not is_mask else Image.new('L', (width, height))

        # 拼接第一个部分
        stitched_image.paste(parts[0], (0, 0))
        left = parts[0].width

        if not is_mask:
            # 对第一个和第二个部分进行插值
            blended_section_1 = Image.blend(
                parts[0].crop((parts[0].width - overlap_width, 0, parts[0].width, height)),
                parts[1].crop((0, 0, overlap_width, height)),
                alpha=0.5
            )
            # 将插值部分和第二部分粘贴到拼接图像
            stitched_image.paste(blended_section_1, (left - overlap_width, 0))
        
        # 将第二部分粘贴到图像
        stitched_image.paste(parts[1], (left, 0))
        left += parts[1].width

        if not is_mask:
            # 对第二个和第三个部分进行插值
            blended_section_2 = Image.blend(
                parts[1].crop((parts[1].width - overlap_width, 0, parts[1].width, height)),
                parts[2].crop((0, 0, overlap_width, height)),
                alpha=0.5
            )
            # 将插值部分和第三部分粘贴到拼接图像
            stitched_image.paste(blended_section_2, (left - overlap_width, 0))
        
        # 将第三部分粘贴到图像
        stitched_image.paste(parts[2], (left, 0))
    
    elif part_num == 2:
        # 对于2个部分的拼接
        left_part = parts[0]
        right_part = parts[1]
        
        width, height = left_part.size[0] + right_part.size[0], left_part.size[1]
        stitched_image = Image.new('RGB', (width, height)) if not is_mask else Image.new('L', (width, height))
        
        # 将左侧图像粘贴在新的图像上
        stitched_image.paste(left_part, (0, 0))
        
        if not is_mask:
            # 使用插值算法，在两张图像的边界处创建过渡
            blended_section = Image.blend(
                left_part.crop((left_part.width - overlap_width, 0, left_part.width, height)),
                right_part.crop((0, 0, overlap_width, height)),
                alpha=0.5
            )
            # 将插值部分粘贴在拼接图像上
            stitched_image.paste(blended_section, (left_part.width - overlap_width, 0))
        
        # 最后将右侧图像粘贴在拼接图像上
        stitched_image.paste(right_part, (left_part.width, 0))
    
    return stitched_image



def process_folder(image_folder, label_folder, output_image_folder, output_label_folder):
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    labels = sorted([f for f in os.listdir(label_folder) if f.endswith('.png')])

    assert len(images) == len(labels), "图像文件和标注文件数量不匹配！"

     # 将图像和标签文件配对后打乱顺序
    image_label_pairs = list(zip(images, labels))
    random.shuffle(image_label_pairs)
    
    # 解压回打乱顺序后的图像和标签列表
    images, labels = zip(*image_label_pairs)

    ensure_dir(output_image_folder)
    ensure_dir(output_label_folder)

    idx = 0

    for img_file, label_file in tqdm(zip(images, labels), total=len(images), desc="Processing files"):
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, label_file)

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        num_parts = random.choice([1,2,3,4])
        if num_parts == 1:
            img.save(os.path.join(output_image_folder, f'stitched_{idx}_{num_parts}_images.jpg'))
            label.save(os.path.join(output_label_folder, f'stitched_{idx}_{num_parts}_images.png'))
            idx += 1
            continue
        stitched_image, stitched_label = crop_images_from_single_image(img, label, num_parts)

        if stitched_image is None or stitched_label is None:
            continue

        stitched_image.save(os.path.join(output_image_folder, f'stitched_{idx}_{num_parts}_images.jpg'))
        stitched_label.save(os.path.join(output_label_folder, f'stitched_{idx}_{num_parts}_images.png'))
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
    cropped_image_four_parts = []
    cropped_mask_four_parts = []

    cropped_image_three_parts = []
    cropped_mask_three_parts = []

    cropped_image_two_parts = []
    cropped_mask_two_parts = []

    image_four_parts_buffer = []
    mask_four_parts_buffer = []

    image_three_parts_buffer = []
    mask_three_parts_buffer = []

    image_two_parts_buffer = []
    mask_two_parts_buffer = []

    dataset_option = 'train'  # 替换为您想要的选项：'train', 'test', 或 'all'
    base_path = "/root/autodl-tmp/standard_project/datasets/datasets_A_flip"  # 数据集的基本路径
    output_path = "/root/autodl-tmp/standard_project/datasets/datasets_A_sliceone"  # 替换为您想要的输出路径
    ensure_dir(output_path)

    process_dataset(base_path, dataset_option, output_path)
