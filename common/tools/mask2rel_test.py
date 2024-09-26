# Description: 将掩码图转换为 RLE 编码格式 图片版

import numpy as np
import os
from PIL import Image
import cv2

def mask_to_rle(mask):
    """
    mask: numpy array, 1 - mask, 0 - background
    Returns run length encoding (RLE) as string format
    """
    pixels = mask.flatten() 
    # This is a very fast way to find the changes in the mask
    use_pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(use_pixels[1:] != use_pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 示例：假设我们有一个简单的掩码
array = np.array([[0,0,0,0,1],[0,1,0,0,1],[1,1,0,0,0]])

image = cv2.imread('000201.png', cv2.IMREAD_GRAYSCALE)
mask = image

rle_encoded = mask_to_rle(array)
print(f'RLE encoded: {rle_encoded}')
