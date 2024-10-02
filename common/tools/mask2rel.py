# Description: 将掩码图转换为 RLE 编码格式 批处理版

import pandas as pd
import os
from PIL import Image
import cv2

def convert_masks_to_rle(mask_folder):
    """
    Convert all masks in a folder to RLE format and save as a CSV.
    """
    data = []
    for filename in os.listdir(mask_folder):
        if filename.endswith(".png"):  # Assuming masks are in .png format
            mask_path = os.path.join(mask_folder, filename)
            mask = np.array(Image.open(mask_path).convert("L"))  # Convert to grayscale
            rle = mask_to_rle(mask)
            image_id = os.path.splitext(filename)[0]
            data.append([image_id, rle])
    
    df = pd.DataFrame(data, columns=["ImageId", "EncodedPixels"])
    df.to_csv("rle_encoded_masks.csv", index=False)
    print("Conversion complete! Saved to rle_encoded_masks.csv")

# 使用方式
mask_folder = 'path_to_mask_folder'
convert_masks_to_rle(mask_folder)
