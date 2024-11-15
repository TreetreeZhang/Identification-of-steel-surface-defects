import cv2
import numpy as np
import random
import os

# 自然过渡800*2=1600张 共计1600张
img_num = 1600

# Paths to the folders containing images and labels
image_folder_path = "/root/autodl-tmp/standard_project/datasets/datasets_A_flip/train/image"    # 这个不要动
label_folder_path = "/root/autodl-tmp/standard_project/datasets/datasets_A_flip/train/label"    # 这个不要动

# Output paths
output_image_folder = "/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train/image"
output_label_folder = "/root/autodl-tmp/standard_project/datasets/double_origin_paste_slice_blend_noise/train/label"
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Get the list of image and label file names in the folders
image_files = os.listdir(image_folder_path)
label_files = os.listdir(label_folder_path)

# Ensure there are enough images and labels
assert len(image_files) >= 2, "Not enough images in the folder to perform the blending."
assert len(label_files) >= 2, "Not enough labels in the folder to perform the blending."

# Function to load random images and labels
def load_random_images_and_labels():
    file1 = random.choice(image_files)
    file2 = random.choice(image_files)
    img1_path = os.path.join(image_folder_path, file1)
    img2_path = os.path.join(image_folder_path, file2) 

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Generate corresponding label paths based on image filenames
    label1_path = os.path.join(label_folder_path, file1.replace(".jpg", ".png"))
    label2_path = os.path.join(label_folder_path, file2.replace(".jpg", ".png"))

    # Load labels
    label1 = cv2.imread(label1_path, cv2.IMREAD_GRAYSCALE)
    label2 = cv2.imread(label2_path, cv2.IMREAD_GRAYSCALE)
    
    return img1, img2, label1, label2

    
    return img1, img2, label1, label2

# Function to blend two images with a smooth transition
def blend_images(img1, img2):
    # Ensure both images are 200x200
    assert img1.shape == (200, 200), f"Image 1 must be 200x200, but got {img1.shape}"
    assert img2.shape == (200, 200), f"Image 2 must be 200x200, but got {img2.shape}"

    # Randomly decide the width of the left and right sections for the images
    width_img1 = random.randint(50, 150)  # Random width for img1 (between 50 and 150)
    width_img2 = 200 - width_img1        # The remaining width for img2 so that total width is 200

    # Get the left part of img1 and the right part of img2
    img1_left = img1[:, :width_img1]   # Left part of img1
    img2_right = img2[:, width_img1:]  # Right part of img2

    # Create a simple linear blend in the overlapping region to achieve a smoother transition
    blend_height = 20  # You can adjust this value

    # Create an empty canvas for the final image
    canvas = np.zeros((200, 200), dtype=img1.dtype)

    # Place the left half of img1 on the canvas
    canvas[:, :width_img1] = img1_left

    # Linear blend between the overlapping region of img1 and img2
    for i in range(blend_height):
        alpha = i / blend_height
        canvas[:, width_img1 + i] = (1 - alpha) * img1[:, width_img1 + i] + alpha * img2[:, width_img1 + i]

    # Place the right half of img2 on the canvas
    canvas[:, width_img1 + blend_height:] = img2_right[:, blend_height:]

    return canvas, width_img1

# Function to directly concatenate labels
def concat_labels(label1, label2, width_img1):
    # Ensure both labels are 200x200
    assert label1.shape == (200, 200), f"Label 1 must be 200x200, but got {label1.shape}"
    assert label2.shape == (200, 200), f"Label 2 must be 200x200, but got {label2.shape}"

    # Randomly decide the width of the left and right sections for the labels
    width_label1 = width_img1  # Random width for label1 (between 50 and 150)
    width_label2 = 200 - width_label1        # The remaining width for label2 so that total width is 200

    # Get the left part of label1 and the right part of label2
    label1_left = label1[:, :width_label1]   # Left part of label1
    label2_right = label2[:, width_label1:]  # Right part of label2

    # Create the canvas for the final image (200x200)
    label_canvas = np.zeros((200, 200), dtype=label1.dtype)

    # Place the left half of label1 on the canvas
    label_canvas[:, :width_label1] = label1_left

    # Place the right half of label2 on the canvas
    label_canvas[:, width_label1:] = label2_right

    return label_canvas

# Generate 5000 images and their labels
for i in range(img_num):
    # Load random images and labels
    img1, img2, label1, label2 = load_random_images_and_labels()
    
    # Blend the images with a smooth transition
    blended_image, width = blend_images(img1, img2)

    # Directly concatenate the labels without blending
    concatenated_label = concat_labels(label1, label2, width)

    # Define output paths for the blended image and label
    output_image_path = os.path.join(output_image_folder, f"blended_image_{i+1}.jpg")
    output_label_path = os.path.join(output_label_folder, f"blended_image_{i+1}.png")

    # Save the blended image and label
    cv2.imwrite(output_image_path, blended_image)
    cv2.imwrite(output_label_path, concatenated_label)

print(f"{img_num} blended images and labels have been saved to the output folder.")
