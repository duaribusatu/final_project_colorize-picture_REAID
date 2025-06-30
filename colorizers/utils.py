from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F

# Normalization constants
L_MEAN = 50.
L_STD = 100.
AB_STD = 110.

def normalize_l(tens_l):
    return (tens_l - L_MEAN) / L_STD

def unnormalize_ab(tens_ab):
    return tens_ab * AB_STD

def preprocess_img(img_rgb_orig, HW=(256, 256), resample=Image.BICUBIC):
    img_rgb_rs = np.asarray(Image.fromarray(img_rgb_orig).resize((HW[1], HW[0]), resample=resample))

    img_lab_orig = color.rgb2lab(img_rgb_orig).astype(np.float32)
    img_lab_rs = color.rgb2lab(img_rgb_rs).astype(np.float32)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.tensor(img_l_orig).unsqueeze(0).unsqueeze(0)
    tens_rs_l = torch.tensor(img_l_rs).unsqueeze(0).unsqueeze(0)

    tens_rs_l = normalize_l(tens_rs_l)

    return tens_orig_l, tens_rs_l

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    out_ab = F.interpolate(out_ab, size=tens_orig_l.shape[2:], mode=mode)
    out_ab = unnormalize_ab(out_ab)

    out_lab = torch.cat([tens_orig_l, out_ab], dim=1)
    out_np = out_lab[0].cpu().numpy().transpose(1, 2, 0)
    out_rgb = color.lab2rgb(out_np)
    return np.clip(out_rgb, 0, 1)
