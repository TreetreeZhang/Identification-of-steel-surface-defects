import shutil
import os

def copy_files(source_folder, target_folder):
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print("源文件夹不存在，请检查路径。")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"目标文件夹已创建：{target_folder}")

    # 遍历源文件夹中的所有文件
    for file_name in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file_name)
        target_file = os.path.join(target_folder, file_name)

        # 如果是文件，则复制
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_file)
            print(f"文件已复制：{file_name}")

        # 如果是文件夹，则递归复制
        elif os.path.isdir(source_file):
            copy_files(source_file, target_file)

# 使用示例
image_folder_src = '/root/autodl-tmp/standard_project/datasets/datasets_B_every5_split/train/image'  # 这个不要变
image_folder_dst = '/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train/image'  # 替换为你的目标文件夹路径

label_folder_src = '/root/autodl-tmp/standard_project/datasets/datasets_B_every5_split/train/label'  # 这个不要变
label_folder_dst = '/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train/label'  # 替换为你的目标文件夹路径

copy_files(image_folder_src, image_folder_dst)
copy_files(label_folder_src, label_folder_dst)