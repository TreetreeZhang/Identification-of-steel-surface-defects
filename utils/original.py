from PIL import Image
import cv2
import numpy as np
import random
import os

from tqdm import tqdm

# 检查mask是否只有除了0以外只有fault_class 
def check_mask(mask, fault_class):
    mask = np.array(mask)
    unique_classes = np.unique(mask)
    return len(unique_classes) == 2 and 0 in unique_classes and fault_class in unique_classes

def check_mask_double(mask):
    mask = np.array(mask)
    unique_classes = np.unique(mask)
    return len(unique_classes) == 3 and 0 in unique_classes

# 处理文件夹中的图片和标注，确保对应
def process_folder(image_folder, annotation_folder,output_image_folder,
                   output_annotation_folder, fault_class=1, image_num=5000):
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    annotations = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.png')])

    assert len(images) == len(annotations), "图像文件和标注文件数量不匹配！"

    # 将图像和标注文件成对配对并随机打乱顺序
    image_annotation_pairs = list(zip(images, annotations))
    random.shuffle(image_annotation_pairs)

    cnt = 1

    for img_file, mask_file in tqdm(image_annotation_pairs, total=len(image_annotation_pairs), desc="Processing files"):
        if cnt > image_num:
            break

        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(annotation_folder, mask_file)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if fault_class == 'double':
            if(check_mask_double(mask)):
                img_name = os.path.basename(img_path)
                mask_name = os.path.basename(mask_path)

                new_img_name = f"original_{cnt}_double_{img_name}"
                new_mask_name = f"original_{cnt}_double_{mask_name}"

                img.save(os.path.join(output_image_folder, new_img_name))
                mask.save(os.path.join(output_annotation_folder, new_mask_name))

                cnt += 1
        else:
            if(check_mask(mask, fault_class)):

                img_name = os.path.basename(img_path)
                mask_name = os.path.basename(mask_path)

                new_img_name = f"original_{cnt}_{fault_class}_{img_name}"
                new_mask_name = f"original_{cnt}_{fault_class}_{mask_name}"

                img.save(os.path.join(output_image_folder, new_img_name))
                mask.save(os.path.join(output_annotation_folder, new_mask_name))

                cnt += 1

if __name__ == "__main__":
    image_folder = "/root/autodl-tmp/standard_project/datasets/datasets_A_flip/train/image"      # 这个不要动
    annotation_folder = "/root/autodl-tmp/standard_project/datasets/datasets_A_flip/train/label"     # 这个不要动
    output_image_folder = "/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train/image"    
    output_annotation_folder = "/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train/label"
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_annotation_folder, exist_ok=True)


    # 第一类缺陷900*2=1800张 第二类缺陷950*2=1900张 第三类缺陷900*2=1800张 双缺陷850*2=1700张 共计7200张
    process_folder(image_folder, annotation_folder, output_image_folder, output_annotation_folder, fault_class='double', image_num=1700)
    process_folder(image_folder, annotation_folder, output_image_folder, output_annotation_folder, fault_class=1, image_num=1800)
    process_folder(image_folder, annotation_folder, output_image_folder, output_annotation_folder, fault_class=2, image_num=1900)
    process_folder(image_folder, annotation_folder, output_image_folder, output_annotation_folder, fault_class=3, image_num=1800)