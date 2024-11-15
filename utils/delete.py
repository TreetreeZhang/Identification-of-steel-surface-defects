import os

# 指定数据集的路径
dataset_path = '/root/autodl-tmp/standard_project/datasets/datasets_slice_paste_noise_blend_1/train'
image_folder = os.path.join(dataset_path, 'image')
label_folder = os.path.join(dataset_path, 'label')

def delete_files_with_fused(folder_path):
    for filename in os.listdir(folder_path):
        if 'fused' in filename:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# 删除 image 和 label 文件夹中名字包含 "fused" 的文件
delete_files_with_fused(image_folder)
delete_files_with_fused(label_folder)

print("All files containing 'fused' in image and label folders have been deleted.")