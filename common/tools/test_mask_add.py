# Description: 本代码实现了将掩码图叠加到原始图像上的功能，即将掩码值为1的区域高亮显示（比如红色）。

import cv2
import numpy as np

# 读取原始图像（假设是彩色的）和掩码图（假设是二值化的灰度图）
original_image = cv2.imread('000201.jpg')  # 读取彩色原始图像
mask_image = cv2.imread('000201.png', cv2.IMREAD_GRAYSCALE)  # 读取灰度掩码图

# 将掩码值扩展为3通道，以匹配原始图像的RGB通道
mask_image_3channel = cv2.merge([mask_image]*3)

# 创建一个高亮效果（比如红色高亮），即将掩码值为1的区域显示为红色
highlighted_image = original_image.copy()
highlighted_image[mask_image == 1] = [0, 0, 255]  # 使掩码为1的地方高亮为红色

# 叠加原始图像和高亮区域
result_image = np.where(mask_image_3channel == 0, original_image, highlighted_image)

# 显示结果图像
cv2.imshow("Result Image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存叠加结果
cv2.imwrite('highlighted_result.png', result_image)
