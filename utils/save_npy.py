import torch
import numpy as np
import os
import time
from PIL import Image, ImageEnhance
import cv2  # 用于图像叠加
# from model.fzx_model import *
# from model.new_try_zzy import *
from model.zzy_model2 import *
from model.unet import UNet
from dataloader import Crack
from tools import *

# def apply_colored_mask(image, mask, color, alpha=0.4):
#     """将带颜色的掩码叠加到原图上"""
#     colored_mask = np.zeros_like(image)
#     colored_mask[mask == 1] = color  # 将掩码中的1区域赋予指定颜色

#     # 将带颜色的掩码和原图按alpha值进行叠加
#     return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

def evaluate(data_path):
    test_loader = torch.utils.data.DataLoader(
        Crack(data_path, 'test'),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)
    
    train_loader = torch.utils.data.DataLoader(
        Crack(data_path, 'train'),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    net = self_net().to(device)
    # net.load_state_dict(torch.load('/root/autodl-tmp/standard_project/FinalModelPth/zzy_model4/0.8356.pth', map_location=device))
    net = torch.load('/root/autodl-tmp/standard_project/FinalModelPth/zzy_model4/0.8365.pth')
    UUnet = torch.load('weights/unet/unet_model_0.7426205467621503.pth')
    # net1 = UNet().to(device)
    # net1.load_state_dict(torch.load('./weights/unet/unet_best_model_20241004_234747.pth', map_location=device))

    net.eval()
    UUnet.eval()

    # Ensure mask folders exist
    os.makedirs('./mask/test_predictions', exist_ok=True)
    os.makedirs('./mask/test_ground_truths', exist_ok=True)
    os.makedirs('./mask/baseline_predictions', exist_ok=True)

    os.makedirs('./submission/test_predictions', exist_ok=True)
    os.makedirs('./submission/test_ground_truths', exist_ok=True)
    os.makedirs('./submission/baseline_predictions', exist_ok=True)

    os.makedirs('./markadd/test_predictions', exist_ok=True)
    os.makedirs('./markadd/test_ground_truths', exist_ok=True)
    os.makedirs('./markadd/baseline_predictions', exist_ok=True)

    with torch.no_grad():
        unet_time = 0 
        pred_time = 0
        unet_img_num = 0
        pred_img_num = 0
        loader = test_loader

        iou = [[],[[],[],[],[]],[[],[],[],[]]]

        for i, (img, lab) in enumerate(loader, start=1):
            img, lab = img.to(device), lab.to(device)
            lab = lab.type(torch.LongTensor)

            # Model predictions
            # baseline_unet
            unet_start_time = time.time()
            pred1 = torch.argmax(UUnet(img).squeeze(0), dim=0, keepdim=True).cpu().numpy()
            unet_end_time = time.time()
            unet_time += (unet_end_time - unet_start_time)
            unet_img_num +=1

            #pred 
            pred_start_time = time.time()
            pred = torch.argmax(net(img).squeeze(0), dim=0, keepdim=True).cpu().numpy()
            pred_end_time = time.time()
            pred_time += (pred_end_time - pred_start_time)
            pred_img_num += 1 
            lab = lab.cpu().numpy()

            # Save predictions and ground truths as .npy
            np.save(f'./submission/test_predictions/prediction_{i:06d}.npy', pred)
            np.save(f'./submission/test_ground_truths/ground_truth_{i:06d}.npy', lab)
            np.save(f'./submission/baseline_predictions/prediction_{i:06d}.npy', pred1)

            # Convert to mask images (grayscale, assuming binary classification)
            pred_mask = Image.fromarray((pred.squeeze(0) * 255).astype(np.uint8))
            pred1_mask = Image.fromarray((pred1.squeeze(0) * 255).astype(np.uint8))
            lab_mask = Image.fromarray((lab.squeeze(0) * 255).astype(np.uint8))

            iou_unet = compute_iou_with_matrix(pred1, lab, 4)
            iou_pred = compute_iou_with_matrix(pred, lab, 4)

            for i in range(1, 4):
                if iou_unet[i]>=0:
                    iou[1][i].append(iou_unet[i])
                if iou_pred[i]>=0:
                    iou[2][i].append(iou_pred[i])

            # Save mask images
            pred_mask.save(f'./mask/test_predictions/mask_{i:06d}.png')
            pred1_mask.save(f'./mask/baseline_predictions/mask_{i:06d}.png')
            lab_mask.save(f'./mask/test_ground_truths/mask_{i:06d}.png')



            # Convert the image tensor to numpy array and rescale to 0-255 for visualization
            img_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            img_np = (img_np * 255).astype(np.uint8)

            # # Create overlay for each mask
            pred_overlay = maskadd(img_np, pred.squeeze(0),alpha=0.2)
            pred1_overlay = maskadd(img_np, pred1.squeeze(0),alpha=0.2)
            lab_overlay = maskadd(img_np, lab.squeeze(0),alpha=0.2)
       
            # Save overlay images
            cv2.imwrite(f'./markadd/test_predictions/overlay_{i:06d}.png', pred_overlay)
            cv2.imwrite(f'./markadd/test_ground_truths/overlay_{i:06d}.png', lab_overlay)
            cv2.imwrite(f'./markadd/baseline_predictions/overlay_{i:06d}.png', pred1_overlay)

        average_time_per_image_unet = unet_time / unet_img_num
        fps_unet = 1 / average_time_per_image_unet
        print (f'Unet FPS:{fps_unet:.2f}')

        average_time_per_image_pred = pred_time / pred_img_num
        fps_pred = 1 / average_time_per_image_pred
        print (f'pred FPS:{fps_pred:.2f}')

        iouunet_1 = np.mean(iou[1][1])
        iouunet_2 = np.mean(iou[1][2])
        iouunet_3 = np.mean(iou[1][3])
        miou_unet = (iouunet_1 + iouunet_2 + iouunet_3) / 3

        ioupred_1 = np.mean(iou[2][1])
        ioupred_2 = np.mean(iou[2][2])
        ioupred_3 = np.mean(iou[2][3])
        miou_pred = (ioupred_1 + ioupred_2 + ioupred_3) / 3

        print(f'Unet MIOU:{miou_unet}')
        print(f'Pred MIOU:{miou_pred}')

if __name__ == "__main__":
    evaluate('./Dataset')

