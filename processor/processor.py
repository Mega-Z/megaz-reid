import logging
import time
import torch
import numpy as np
import cv2
import os
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
from timm.data.random_erasing import RandomErasing
import torch.distributed as dist
from log_handler.json_logger import JsonLogger
from dataset_stat import DataStat

from detectron2.model_zoo import get_config
from detectron2.engine.defaults import DefaultPredictor


def patchify(imgs):
    """
	imgs: (N, 3, H, W)
	x: (N, L, patch_size**2 *3)
	"""
    p = 16
    assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x


def unpatchify(x, H, W):
    """
	x: (N, L, patch_size**2 *3)
	imgs: (N, 3, H, W)
	"""
    p = 16
    h = H // p
    w = W // p
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, H, W))
    return imgs


def patchify_rgb_avg(imgs, avg_size=16):
    """
	imgs: (N, 3, H, W)
	x: (N, L, (16/a)**2*3)
	"""
    p = 16
    a = avg_size
    assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p//a, a, w, p//a, a))
    x = torch.einsum('nchpawqb->nhwabpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, a**2, (p//a)**2*3))
    x = torch.mean(x, 2)
    return x


def unpatchify_rgb_avg(x, H, W, avg_size):
    """
	x: (N, L, (16/a)**2*3)
	imgs: (N, 3, H, W)
	"""
    p = 16
    a = avg_size
    h = H // p
    w = W // p
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p//a, 1, p//a, 1, 3))
    x = x.repeat(1, 1, 1, 1, a, 1, a, 1)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, H, W))
    return imgs


def unpatchify_mask(m, H, W):
    """
    Args:
        m: (N, L)
        H: int
        W: int

    Returns:
        mask: (N, 3, H, W)
    """
    p = 16
    h = H // p
    w = W // p
    assert h * w == m.shape[1]

    m = m.reshape(shape=(m.shape[0], h, w, 1, 1, 1))
    m = m.repeat(1, 1, 1, p, p, 3)
    m = torch.einsum('nhwpqc->nchpwq', m)
    mask = m.reshape(shape=(m.shape[0], 3, H, W))
    return mask


def occlude(img, vid, min_occ_ratio=0.2):
    out = img.clone()
    for idx in range(img.shape[0]):
        # choose an image from different id to occlude im
        idx_occ = torch.randint(low=0, high=vid.shape[0]-1, size=(1,))[0]
        while vid[idx] == vid[idx_occ]:
            idx_occ = torch.randint(low=0, high=vid.shape[0]-1, size=(1,))[0]
        # randomize the occluded region
        x_1 = torch.randint(low=0, high=int(img.shape[3]*(1-min_occ_ratio))-1, size=(1,))[0]
        y_1 = torch.randint(low=0, high=int(img.shape[2]*(1-min_occ_ratio))-1, size=(1,))[0]
        x_2 = torch.randint(low=x_1, high=img.shape[3], size=(1,))[0]
        y_2 = torch.randint(low=y_1, high=img.shape[2], size=(1,))[0]
        while x_2-x_1 < img.shape[3]*min_occ_ratio or y_2-y_1 < img.shape[2]*min_occ_ratio:
            x_1 = torch.randint(low=0, high=int(img.shape[3]*(1-min_occ_ratio))-1, size=(1,))[0]
            y_1 = torch.randint(low=0, high=int(img.shape[2]*(1-min_occ_ratio))-1, size=(1,))[0]
            x_2 = torch.randint(low=x_1, high=img.shape[3], size=(1,))[0]
            y_2 = torch.randint(low=y_1, high=img.shape[2], size=(1,))[0]
        out[idx, :, y_1:y_2, x_1:x_2] = img[idx_occ, :, y_1:y_2, x_1:x_2]
    return out


def occlude_constrained(img, vid, align_bottom=True, min_occ_ratio=(0.2, 0.2), max_occ_ratio=(1.0, 0.5)):
    out = img.clone()
    for idx in range(img.shape[0]):
        # choose an image from different id to occlude im
        idx_occ = torch.randint(low=0, high=vid.shape[0] - 1, size=(1,))[0]
        while vid[idx] == vid[idx_occ]:
            idx_occ = torch.randint(low=0, high=vid.shape[0] - 1, size=(1,))[0]
        # randomize the occluded region
        x_1 = torch.randint(low=0, high=int(img.shape[3] * (1 - min_occ_ratio[0])) - 1, size=(1,))[0]
        y_1 = torch.randint(low=0, high=int(img.shape[2] * (1 - min_occ_ratio[1])) - 1, size=(1,))[0]
        x_2 = torch.randint(low=x_1, high=img.shape[3], size=(1,))[0]
        if align_bottom:
            y_2 = img.shape[2]-1
        else:
            y_2 = torch.randint(low=y_1, high=img.shape[2], size=(1,))[0]
        while not(img.shape[3]*min_occ_ratio[0] < x_2-x_1 < img.shape[3]*max_occ_ratio[0] and img.shape[2]*min_occ_ratio[1] < y_2-y_1 < img.shape[2]*max_occ_ratio[1]):
            x_1 = torch.randint(low=0, high=int(img.shape[3] * (1 - min_occ_ratio[0])) - 1, size=(1,))[0]
            y_1 = torch.randint(low=0, high=int(img.shape[2] * (1 - min_occ_ratio[1])) - 1, size=(1,))[0]
            x_2 = torch.randint(low=x_1, high=img.shape[3], size=(1,))[0]
            if align_bottom:
                y_2 = img.shape[2]-1
            else:
                y_2 = torch.randint(low=y_1, high=img.shape[2], size=(1,))[0]
        shift_x = torch.randint(low=0 - x_1, high=img.shape[3] - x_2, size=(1,))[0]
        shift_y = torch.randint(low=0 - y_1, high=img.shape[2] - y_2, size=(1,))[0]
        out[idx, :, y_1:y_2, x_1:x_2] = img[idx_occ, :, y_1+shift_y:y_2+shift_y, x_1+shift_x:x_2+shift_x]
    return out


def rect_mask_patch_align(w, h, min_occ_ratio=(0.2, 0.2), max_occ_ratio=(1.0, 0.5), margin_top=0.3, align_bottom=True):
    patch_size = 16
    pw = w // patch_size
    ph = h // patch_size
    mask = torch.zeros((h, w), dtype=torch.float32)
    patch_mask = torch.zeros((ph, pw), dtype=torch.float32)
    x1 = torch.randint(low=0, high=int(pw*(1-min_occ_ratio[0])) + 1, size=(1,))[0]
    x2 = torch.randint(low=x1+int(min_occ_ratio[0]*pw), high=min(x1+int(max_occ_ratio[0]*pw), pw) + 1, size=(1,))[0]
    if align_bottom:
        y1 = torch.randint(low=int((1-max_occ_ratio[1])*ph), high=int((1-min_occ_ratio[1])*ph), size=(1,))[0]
        y2 = ph
    else:
        y1 = torch.randint(low=int(margin_top * ph), high=int(ph*(1-min_occ_ratio[1])) + 1, size=(1,))[0]
        y2 = torch.randint(low=y1+int(min_occ_ratio[1]*pw), high=min(y1+int(max_occ_ratio[1]*ph), ph) + 1, size=(1,))[0]
    bound = [x1 * patch_size, x2 * patch_size, y1 * patch_size, y2 * patch_size]
    patch_mask[y1:y2, x1:x2] = 1
    mask[bound[2]:bound[3], bound[0]:bound[1]] = 1
    mask_bias = torch.zeros_like(mask)
    x_bias = torch.randint(low=0 - bound[0], high=w + 1 - bound[1], size=(1,))[0]
    y_bias = torch.randint(low=0 - bound[2], high=h + 1 - bound[3], size=(1,))[0]
    mask_bias[bound[2] + y_bias:bound[3] + y_bias, bound[0] + x_bias:bound[1] + x_bias] = mask[bound[2]:bound[3], bound[0]:bound[1]]
    # print(mask.mean(), mask_bias.mean())
    patch_mask = torch.flatten(patch_mask)
    return mask == 1, mask_bias == 1, patch_mask


def blockwise_mask(w, h, masking_ratio=[0.2], min_size=64, aspect_ratio=0.3, margin_top=0.3):
    # TODO: add align_bottom option
    mask = torch.zeros((h, w), dtype=torch.float32)
    bound = [w-1, 0, h-1, 0]  # x1, x2, y1, y2
    if len(masking_ratio) == 2:  # [min, max]
        m_ratio = masking_ratio[0] + torch.rand(1)*masking_ratio[1]-masking_ratio[0]
    else:
        m_ratio = masking_ratio[0]
    while mask.mean() < m_ratio and min_size < int(np.ceil(h*w*(m_ratio-mask.mean()))):
        block_size = torch.randint(low=min_size, high=int(np.ceil(h*w*(m_ratio-mask.mean()))), size=(1,))[0]
        block_w, block_h = -1, -1
        while not (0 < block_w < w and 0 < block_h < h - int(margin_top*h)):
            flag = torch.rand(size=(1,))[0] > 0.5
            ratio = (1-torch.rand(size=(1,))*(1-aspect_ratio)) if flag else 1/(1-torch.rand(size=(1,))*(1-aspect_ratio))  # [0.3, 1/0.3]
            block_w = int(np.ceil(np.sqrt(block_size*ratio)))
            block_h = int(np.ceil(np.sqrt(block_size/ratio)))
        x1 = torch.randint(low=0, high=w - block_w, size=(1, ))[0]
        y1 = torch.randint(low=int(margin_top*h), high=h - block_h, size=(1, ))[0]
        mask[y1:y1+block_h, x1:x1+block_w] = 1
        # update boundary
        bound[0] = min(x1, bound[0])
        bound[1] = max(x1+block_w, bound[1])
        bound[2] = min(y1, bound[2])
        bound[3] = max(y1+block_h, bound[3])
    mask_bias = torch.zeros_like(mask)
    x_bias = torch.randint(low=0 - bound[0], high=w - 1 - bound[1], size=(1,))[0] if bound[1] - bound[0] != w - 1 else 0
    y_bias = torch.randint(low=0 - bound[2], high=h - 1 - bound[3], size=(1,))[0] if bound[3] - bound[2] != h - 1 else 0
    mask_bias[bound[2]+y_bias:bound[3]+y_bias, bound[0]+x_bias:bound[1]+x_bias] = mask[bound[2]:bound[3], bound[0]:bound[1]]
    return mask == 1, mask_bias == 1


def blockwise_mask_patch_align(w, h, masking_ratio=[0.2], min_size=4, aspect_ratio=0.3, margin_top=0.3, align_bottom=True):
    patch_size = 16
    pw = w // patch_size
    ph = h // patch_size
    mask = torch.zeros((h, w), dtype=torch.float32)
    patch_mask = torch.zeros((ph, pw), dtype=torch.float32)
    bound = [w, 0, h, 0]  # x1, x2, y1, y2
    if len(masking_ratio) == 2:  # [min, max]
        m_ratio = masking_ratio[0] + torch.rand(1)*masking_ratio[1]-masking_ratio[0]
    else:
        m_ratio = masking_ratio[0]
    while patch_mask.mean() < m_ratio and min_size < int(np.ceil(ph*pw*(m_ratio-patch_mask.mean()))):
        block_size = torch.randint(low=min_size, high=int(ph*pw*(m_ratio-patch_mask.mean()))+1, size=(1,))[0]
        block_w, block_h = -1, -1
        while not (0 < block_w < pw and 0 < block_h < ph - int(margin_top*ph)):
            hw_flag = torch.rand(size=(1,))[0] > 0.5
            ratio = (1-torch.rand(size=(1,))*(1-aspect_ratio)) if hw_flag else 1/(1-torch.rand(size=(1,))*(1-aspect_ratio))  # [0.3, 1/0.3]
            block_w = int(np.ceil(np.sqrt(block_size*ratio)))
            block_h = int(np.ceil(np.sqrt(block_size/ratio)))
        x1 = torch.randint(low=0, high=pw - block_w + 1, size=(1, ))[0]
        y1 = torch.randint(low=int(margin_top*ph), high=ph - block_h + 1, size=(1, ))[0]
        patch_mask[y1:y1+block_h, x1:x1+block_w] = 1
        mask[y1*patch_size:(y1+block_h)*patch_size, x1*patch_size:(x1+block_w)*patch_size] = 1
        # update boundary
        bound[0] = min(x1*patch_size, bound[0])
        bound[1] = max((x1+block_w)*patch_size, bound[1])
        bound[2] = min(y1*patch_size, bound[2])
        bound[3] = max((y1+block_h)*patch_size, bound[3])
    if align_bottom:
        if bound[3] < h:
            mask_ = torch.zeros_like(mask)
            patch_mask_ = torch.zeros_like(patch_mask)
            y_shift = h - bound[3]
            mask_[bound[2] + y_shift:bound[3]+y_shift] = mask[bound[2]:bound[3]]
            mask = mask_
            patch_mask_[(bound[2]+y_shift)//patch_size:(bound[3]+y_shift)//patch_size] = patch_mask[bound[2]//patch_size:bound[3]//patch_size]
            patch_mask = patch_mask_
            bound[2] += y_shift
            bound[3] += y_shift
    mask_bias = torch.zeros_like(mask)
    x_bias = torch.randint(low=0 - bound[0], high=w + 1 - bound[1], size=(1,))[0]
    y_bias = torch.randint(low=0 - bound[2], high=h + 1 - bound[3], size=(1,))[0]
    mask_bias[bound[2]+y_bias:bound[3]+y_bias, bound[0]+x_bias:bound[1]+x_bias] = mask[bound[2]:bound[3], bound[0]:bound[1]]
    patch_mask = torch.flatten(patch_mask)
    return mask == 1, mask_bias == 1, patch_mask


def blockwise_mask_patch_align_udlr(w, h, masking_ratio=[0.2], min_size=4, aspect_ratio=0.3, margin=0.3, ulrd=[0.25, 0.25, 0.25, 0.25], align_bound=True):
    patch_size = 16
    pw = w // patch_size
    ph = h // patch_size
    mask = torch.zeros((h, w), dtype=torch.float32)
    patch_mask = torch.zeros((ph, pw), dtype=torch.float32)
    bound = [w, 0, h, 0]  # x1, x2, y1, y2
    if len(masking_ratio) == 2:  # [min, max]
        m_ratio = masking_ratio[0] + torch.rand(1)*masking_ratio[1]-masking_ratio[0]
    else:
        m_ratio = masking_ratio[0]
    ulrd_flag = torch.rand(size=(1,))[0]
    if ulrd_flag <= ulrd[0]:
        direction = "up"
    elif ulrd_flag <= ulrd[0] + ulrd[1]:
        direction = "left"
    elif ulrd_flag <= ulrd[0] + ulrd[1] + ulrd[2]:
        direction = "right"
    else:
        direction = "down"
    while patch_mask.mean() < m_ratio and min_size < int(np.ceil(ph*pw*(m_ratio-patch_mask.mean()))):
        block_size = torch.randint(low=min_size, high=int(ph*pw*(m_ratio-patch_mask.mean()))+1, size=(1,))[0]
        block_w, block_h = -1, -1
        # generate block
        while not ((0 < block_w < pw and 0 < block_h < ph - int(margin*ph) and (direction == "up" or direction == "down")) or
                   (0 < block_w < pw - int(margin*pw) and 0 < block_h < ph and (direction == "left" or direction == "right"))):
            hw_flag = torch.rand(size=(1,))[0] > 0.5
            ratio = (1-torch.rand(size=(1,))*(1-aspect_ratio)) if hw_flag else 1/(1-torch.rand(size=(1,))*(1-aspect_ratio))  # [0.3, 1/0.3]
            block_w = int(np.ceil(np.sqrt(block_size*ratio)))
            block_h = int(np.ceil(np.sqrt(block_size/ratio)))
        # choose position
        if direction == "up":
            x1 = torch.randint(low=0, high=pw - block_w + 1, size=(1, ))[0]
            y1 = torch.randint(low=0, high=ph - block_h + 1 - int(margin*ph), size=(1, ))[0]
        elif direction == "left":
            x1 = torch.randint(low=0, high=pw - block_w + 1 - int(margin*pw), size=(1, ))[0]
            y1 = torch.randint(low=0, high=ph - block_h + 1, size=(1, ))[0]
        elif direction == "right":
            x1 = torch.randint(low=int(margin*pw), high=pw - block_w + 1, size=(1, ))[0]
            y1 = torch.randint(low=0, high=ph - block_h + 1, size=(1, ))[0]
        else:
            x1 = torch.randint(low=0, high=pw - block_w + 1, size=(1, ))[0]
            y1 = torch.randint(low=int(margin*ph), high=ph - block_h + 1, size=(1, ))[0]
        patch_mask[y1:y1+block_h, x1:x1+block_w] = 1
        mask[y1*patch_size:(y1+block_h)*patch_size, x1*patch_size:(x1+block_w)*patch_size] = 1
        # update boundary
        bound[0] = min(x1*patch_size, bound[0])
        bound[1] = max((x1+block_w)*patch_size, bound[1])
        bound[2] = min(y1*patch_size, bound[2])
        bound[3] = max((y1+block_h)*patch_size, bound[3])
    if align_bound:
        if direction == "up" and bound[2] > 0:
            x_shift = 0
            y_shift = 0 - bound[2]
        elif direction == "left" and bound[0] > 0:
            x_shift = 0 - bound[0]
            y_shift = 0
        elif direction == "right" and bound[1] < w:
            x_shift = w - bound[1]
            y_shift = 0
        elif direction == "down" and bound[3] < h:
            x_shift = 0
            y_shift = h - bound[3]
        else:
            x_shift = 0
            y_shift = 0
        mask = torch.roll(mask, [y_shift, x_shift], [0, 1])
        patch_mask = torch.roll(patch_mask, [y_shift//patch_size, x_shift//patch_size], [0, 1])
        bound[0] += x_shift
        bound[1] += x_shift
        bound[2] += y_shift
        bound[3] += y_shift
    mask_bias = torch.zeros_like(mask)
    x_bias = torch.randint(low=0 - bound[0], high=w + 1 - bound[1], size=(1,))[0]
    y_bias = torch.randint(low=0 - bound[2], high=h + 1 - bound[3], size=(1,))[0]
    mask_bias[bound[2]+y_bias:bound[3]+y_bias, bound[0]+x_bias:bound[1]+x_bias] = mask[bound[2]:bound[3], bound[0]:bound[1]]
    patch_mask = torch.flatten(patch_mask)
    return mask == 1, mask_bias == 1, patch_mask


def gen_img_occ(imgs, vids, cfg, idx, img, mask, occ_type, img_id_masks=None):
    h = imgs.shape[2]
    w = imgs.shape[3]
    if occ_type == "instance_mask":
        transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
        mask_resize = T.Resize((h//16,w//16))
        # select a mask from different id
        other_ids = [k for k in img_id_masks.keys() if k!=vids[idx]]
        if len(other_ids) == 0:
            return
        other_id = np.random.choice(other_ids)
        mask_dict = np.random.choice(img_id_masks[other_id])
        instance_img = Image.open(mask_dict["path"]).convert('RGB')
        instance_img = transforms(instance_img).to("cuda")
        instance_mask = mask_dict["mask"]
        img_mask = torch.zeros_like(instance_mask)
        # random x&y shift
        dx = torch.randint(low=0, high=w, size=(1,))[0] - int(w/2)
        dy = torch.randint(low=int(h/4), high=int(h/4*3), size=(1,))[0]
        ori_x1 = max(0, dx)
        ori_x2 = min(w, w+dx)
        ori_y1 = max(0, dy)
        ori_y2 = min(h, h+dy)
        msk_x1 = max(0, -1*dx)
        msk_x2 = min(w, w-dx)
        msk_y1 = max(0, -1*dy)
        msk_y2 = min(h, h-dy)
        m = img_mask[ori_y1:ori_y2, ori_x1:ori_x2] = instance_mask[msk_y1:msk_y2, msk_x1:msk_x2]
        img[:, ori_y1:ori_y2, ori_x1:ori_x2][:, m] = instance_img[:, msk_y1:msk_y2, msk_x1:msk_x2][:, m]
        img_mask = mask_resize(torch.unsqueeze(img_mask,0))
        mask[:] = torch.reshape(img_mask, mask.shape)
    elif occ_type == "img_block":
        idx_occ = torch.randint(low=0, high=vids.shape[0] - 1, size=(1,))[0]
        while vids[idx] == vids[idx_occ]:
            idx_occ = torch.randint(low=0, high=vids.shape[0] - 1, size=(1,))[0]
        mask1, mask2, mask[:] = blockwise_mask_patch_align_udlr(w, h, masking_ratio=cfg.MODEL.OCC_RATIO, ulrd=cfg.MODEL.OCC_ULRD, margin=cfg.MODEL.OCC_MARGIN, align_bound=cfg.MODEL.OCC_ALIGN_BOUND)
        img[:, mask1] = imgs[idx_occ, :, mask2]
    elif occ_type == "img_rect":
        idx_occ = torch.randint(low=0, high=vids.shape[0] - 1, size=(1,))[0]
        while vids[idx] == vids[idx_occ]:
            idx_occ = torch.randint(low=0, high=vids.shape[0] - 1, size=(1,))[0]
        mask1, mask2, mask[:] = rect_mask_patch_align(w, h, align_bottom=cfg.MODEL.OCC_ALIGN_BTM)
        img[:, mask1] = imgs[idx_occ, :, mask2]


def gen_occ(imgs, vids, cfg, id_img_masks=None):
    out = imgs.clone()
    patch_mask = torch.zeros((imgs.shape[0], imgs.shape[2]*imgs.shape[3]//16//16))
    occ_types = cfg.MODEL.OCC_TYPES
    occ_ratios = cfg.MODEL.OCC_TYPES_RATIO
    assert len(occ_types) == len(occ_ratios)
    for idx in range(imgs.shape[0]):
        occ_type_rand = torch.rand(size=(1,))[0]
        occ_ratio_sum = 0
        for i, occ_type in enumerate(occ_types):
            occ_ratio_sum += occ_ratios[i]
            if occ_type_rand <= occ_ratio_sum:
                gen_img_occ(imgs, vids, cfg, idx, out[idx], patch_mask[idx], occ_type, id_img_masks)
                break

    return out, patch_mask == 1

'''
def gen_occ(img, vid, cfg, patch_align=False, id_img_masks=None):
    out = img.clone()
    patch_mask = torch.zeros((img.shape[0], img.shape[2]*img.shape[3]//16//16))
    for idx in range(img.shape[0]):
        # choose an image from different id to occlude im
        idx_occ = torch.randint(low=0, high=vid.shape[0] - 1, size=(1,))[0]
        while vid[idx] == vid[idx_occ]:
            idx_occ = torch.randint(low=0, high=vid.shape[0] - 1, size=(1,))[0]
        if patch_align:
            if cfg.MODEL.OCC_TYPE == "img_block":
                # mask1, mask2, patch_mask[idx] = blockwise_mask_patch_align(img.shape[3], img.shape[2], masking_ratio=cfg.MODEL.OCC_RATIO, align_bottom=cfg.MODEL.OCC_ALIGN_BTM)
                mask1, mask2, patch_mask[idx] = blockwise_mask_patch_align_udlr(img.shape[3], img.shape[2], masking_ratio=cfg.MODEL.OCC_RATIO, ulrd=cfg.MODEL.OCC_ULRD, margin=cfg.MODEL.OCC_MARGIN, align_bound=cfg.MODEL.OCC_ALIGN_BOUND)
            elif cfg.MODEL.OCC_TYPE == "img_rect":
                mask1, mask2, patch_mask[idx] = rect_mask_patch_align(img.shape[3], img.shape[2], align_bottom=cfg.MODEL.OCC_ALIGN_BTM)
        else:
            mask1, mask2 = blockwise_mask(img.shape[3], img.shape[2], masking_ratio=cfg.MODEL.OCC_RATIO)
        out[idx, :, mask1] = img[idx_occ, :, mask2]
    if patch_align:
        return out, patch_mask == 1
    else:
        return out


def gen_instance_occ(img, vid, img_id_masks, cfg):
    out = img.clone()
    h = img.shape[2]
    w = img.shape[3]
    patch_mask = torch.zeros((img.shape[0], h*w//16//16))
    transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    mask_resize = T.Resize((h//16,w//16))
    for idx in range(img.shape[0]):
        if torch.rand(size=(1,))[0] > np.average(cfg.MODEL.OCC_RATIO):
            continue
        # select a mask from different id
        other_ids = [k for k in img_id_masks.keys() if k!=vid[idx]]
        if len(other_ids) == 0:
            continue
        other_id = np.random.choice(other_ids)
        mask_dict = np.random.choice(img_id_masks[other_id])
        instance_img = Image.open(mask_dict["path"]).convert('RGB')
        instance_img = transforms(instance_img).to("cuda")
        instance_mask = mask_dict["mask"]
        img_mask = torch.zeros_like(instance_mask)
        # random x&y shift
        dx = torch.randint(low=0, high=w, size=(1,))[0] - int(w/2)
        dy = torch.randint(low=int(h/4), high=int(h/4*3), size=(1,))[0]
        ori_x1 = max(0, dx)
        ori_x2 = min(w, w+dx)
        ori_y1 = max(0, dy)
        ori_y2 = min(h, h+dy)
        msk_x1 = max(0, -1*dx)
        msk_x2 = min(w, w-dx)
        msk_y1 = max(0, -1*dy)
        msk_y2 = min(h, h-dy)
        m = img_mask[ori_y1:ori_y2, ori_x1:ori_x2] = instance_mask[msk_y1:msk_y2, msk_x1:msk_x2]
        out[idx, :, ori_y1:ori_y2, ori_x1:ori_x2][:, m] = instance_img[:, msk_y1:msk_y2, msk_x1:msk_x2][:, m]
        img_mask = mask_resize(torch.unsqueeze(img_mask,0))
        patch_mask[idx] = torch.reshape(img_mask, patch_mask[idx].shape)

    return out, patch_mask
'''

def shift(img, max_shift_ratio=0.33):
    out = torch.zeros_like(img)
    w = img.shape[3]
    h = img.shape[2]
    shift_direc = torch.rand(img.shape[0])
    shift_ratio = torch.rand(img.shape[0])
    for idx in range(img.shape[0]):
        if shift_direc[idx] < 0.25:  # shift up
            dist = int(shift_ratio[idx]*max_shift_ratio*h)
            out[idx, :, 0:h-dist-1, :] = img[idx, :, dist:h-1, :]
        elif shift_direc[idx] < 0.5:  # shift right
            dist = int(shift_ratio[idx]*max_shift_ratio*w)
            out[idx, :, :, dist:w-1] = img[idx, :, :, 0:w-dist-1]
        elif shift_direc[idx] < 0.75:  # shift down
            dist = int(shift_ratio[idx]*max_shift_ratio*h)
            out[idx, :, dist:h-1, :] = img[idx, :, 0:h-dist-1, :]
        else:  # shift left
            dist = int(shift_ratio[idx]*max_shift_ratio*w)
            out[idx, :, :, 0:w-dist-1] = img[idx, :, :, dist:w-1]
    return out


def rollout(attentions, discard_ratio, head_fusion, occ_token=False):
    """

    Args:
        attentions: [blocks*[B, H, 2+P, 2+P]]
        discard_ratio:
        head_fusion:

    Returns:
        mask: [B, P]
    """
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions: # blocks
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise Exception("Attention head fusion type Not supported")

            # Drop the lowest attentions, but
            # don't drop the class token
            '''
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            '''
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            for i in range(flat.shape[0]):
                _, indices = flat[i].topk(int(flat[i].size(-1)*discard_ratio), -1, False)
                indices = indices[indices != 0]
                indices = indices[indices != 1]
                flat[i, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a_sum = a.sum(dim=-1).unsqueeze(-1)
            a = a / a_sum

            result = torch.matmul(a, result)
    mask = result[:, 0, -128:]
    '''
    if occ_token:
        mask = result[:, 0, 2:]
    else:
        mask = result[:, 0, 1:]
        '''
    '''
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)'''
    return mask


def hook_out_attention(attns_hook):
    """

    Args:
        attns_hook: [blocks*[B, H, 1+P, 1+P]]

    Returns:
        res: [B, H, P]
    """
    result = torch.eye(attns_hook[0].size(-1))
    with torch.no_grad():
        for attn_layer in attns_hook:
            result = torch.matmul(attn_layer, result)
    res = result[:, :, 0, -128:]

    return res.to("cuda")

    # return attns_hook[-1][:, :, 0, -128:] # last layer


def visualize_matches(idx5, val_dataset, path_list, root, id_annotation=False):
    query_root = "../../datasets/Occluded_Duke/query/"
    gallery_root = "../../datasets/Occluded_Duke/bounding_box_test/"
    num_query = idx5.shape[0]
    for q_idx in range(num_query):
        q_id = val_dataset[q_idx][1]
        q_img = cv2.resize(cv2.imread(query_root+path_list[q_idx]), (128, 256))
        g_ids = [val_dataset[num_query+g_idx][1] for g_idx in idx5[q_idx]]
        g_imgs = [cv2.resize(cv2.imread(gallery_root+path_list[num_query+g_idx]), (128, 256)) for g_idx in idx5[q_idx]]
        if id_annotation:
            output = np.ones((300, 868, 3)) * 255
        else:
            output = np.ones((256, 868, 3)) * 255
        cv2.putText(output, "{:0>4d}".format(q_id), (20, 295), 0, 1, (255, 255, 255))
        for m_idx in range(5):
            if id_annotation:
                cv2.putText(output, "{:0>4d}".format(g_ids[m_idx]), (128*(m_idx+1)+20, 295), 0, 1, (255, 255, 255))
            if g_ids[m_idx] == q_id:
                cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 255, 0), 4)
            else:
                cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 0, 255), 4)

        output[:256, :128] = q_img
        output[:256, 228:] = np.concatenate(g_imgs, axis=1)
        cv2.imwrite(root+'/match_results/{:0>5d}.jpg'.format(q_idx+1), output)


def draw_patch_attn(patchembed, H=256, W=128):
    """

    Args:
        patchembed: (B, (1+P), D)
        H:
        W:

    Returns:
        attn_maps: (B, 3, H, W)

    """
    resize = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)
    patch_size = 16
    cls_token = patchembed[:, 0, :].unsqueeze(1)
    image_embed = patchembed[:, 1:, :]
    attn = (cls_token @ image_embed.transpose(-2, -1))
    attn = attn.softmax(dim=-1).squeeze(1)
    max = attn.max(dim=1)[0].unsqueeze(1)
    norm_attn = attn / max * 2 -1
    attn_maps = norm_attn.reshape(shape=(norm_attn.shape[0], H//patch_size, W//patch_size)).unsqueeze(1).repeat(1,3,1,1)
    attn_maps = resize(attn_maps)
    return attn_maps


def visualize_occ_label(patch_mask, H=256, W=128):
    """

    Args:
        occ_pred: (B, P, 2)

    Returns:
        occ_map: (B, 3, H, W)

    """
    resize = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)
    patch_size = 16
    occ_label = patch_mask.float() * 2 - 1
    occ_label = occ_label.reshape(shape=(occ_label.shape[0], H//patch_size, W//patch_size)).unsqueeze(1).repeat(1,3,1,1)
    occ_label = resize(occ_label)
    return occ_label


def visualize_occ_pred(occ_pred, H=256, W=128):
    """

    Args:
        occ_pred: (B, P, 2)

    Returns:
        occ_map: (B, 3, H, W)

    """
    resize = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)
    patch_size = 16
    occ_pred = occ_pred.softmax(dim=-1)
    occ_pred = occ_pred[:, :, 1] * 2 - 1
    occ_map = occ_pred.reshape(shape=(occ_pred.shape[0], H//patch_size, W//patch_size)).unsqueeze(1).repeat(1,3,1,1)
    occ_map = resize(occ_map)
    return occ_map


def generate_masks(train_loader, score_thre=0.6, area_thre=1024, inter_thre=196):
    cfg = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    seg_pred = DefaultPredictor(get_config(cfg, trained=True))
    id_img_masks = {}
    for n_iter, (_, pids, _, _, img_paths) in enumerate(train_loader):
        if (n_iter+1) % 10 == 0:
            print("preparing person masks: ", n_iter+1, "/", len(train_loader))
            # break # for debug
        for i in range(len(img_paths)):
            cv_img = cv2.imread(img_paths[i])
            inputs = cv2.resize(cv_img, (128, 256))
            outputs = seg_pred(inputs)

            # filter masks
            predictions = outputs["instances"]
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
            masks = predictions.pred_masks if predictions.has("pred_masks") else None
            if masks is not None:
                masks = [masks[i] for i in range(len(masks)) if scores[i] >= score_thre and classes[i] == 0]
                # print("old length: ", len(masks), end='')
                # combine masks that has intersection
                del_flag = [False] * len(masks)
                for j in range(len(masks) - 1):
                    for k in range(j + 1, len(masks)):
                        if not del_flag[k]:
                            inter = masks[j] & masks[k]
                            if inter.sum() >= inter_thre:
                                masks[j] = masks[j] | masks[k]
                                del_flag[k] = True
                masks = [masks[j] for j in range(len(masks)) if not del_flag[j] and masks[j].sum() >= area_thre]
                if len(masks)==0:
                    continue
                max_mask = masks[0]
                for j in range(1, len(masks)):
                    if masks[j].sum()>max_mask.sum():
                        max_mask = masks[j]
                # save masks
                mask_dict = {
                    "mask": max_mask,
                    "path": img_paths[i]
                }
                if pids[i].item() not in id_img_masks.keys():
                    id_img_masks.update({
                        pids[i].item():[mask_dict]
                    })
                else:
                    id_img_masks[pids[i].item()].append(mask_dict)
    return id_img_masks


def update_sample_head_salience(sample_head_salience, path, ch_attn):
    B, C = ch_attn.shape
    H = 12
    head_attn = torch.mean(ch_attn.reshape(B, H, -1), dim=2)
    for i,p in enumerate(path):
        sample_head_salience.update({path[i]: head_attn[i]})

def query_sample_head_salience(sample_head_sailence, path):
    pass



def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_json = False
    hook_attn_from_model = False
    iter_save_log = 1
    H = 12

    logger = logging.getLogger("transreid.train")
    if log_json and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
        json_logger = JsonLogger(cfg=cfg)
        logger.info('start training')
        max_rank1 = 0
    _LOCAL_PROCESS_GROUP = None

    if "instance_mask" in cfg.MODEL.OCC_TYPES :
        id_img_masks = generate_masks(train_loader) # each gpu owns one
    else:
        id_img_masks = None
    randomerease = RandomErasing(probability=1, mode='pixel', max_count=1, device='cpu')
    # model = torch.nn.parallel.DataParallel(model, device_ids=[0,1])

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if hook_attn_from_model:
        # append hook
        attns_hook = [] # raw data hook out from model
        modules = {}
        def get_attn(module, input, output):
            attns_hook.append(output.cpu())
        for name, module in model.named_modules():
            if 'attn_drop' in name and not 'occ_block' in name:
                module.register_forward_hook(get_attn)
                modules[name] = module

    if cfg.MODEL.SAMPLE_HEAD_SUP:
        sample_head_salience = {}
    loss_meter = AverageMeter()
    loss_ifrc_meter = AverageMeter()
    loss_occ_meter = AverageMeter()
    loss_id_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
    loss_id_dec_meter = AverageMeter()
    loss_tri_dec_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    #
    paras = {}
    for idx, para in enumerate(model.named_parameters()):
        paras[idx] = para

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_ifrc_meter.reset()
        loss_occ_meter.reset()
        loss_id_meter.reset()
        loss_tri_meter.reset()
        loss_id_dec_meter.reset()
        loss_tri_dec_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        # freeze occ pred
        if cfg.MODEL.OCC_AWARE and epoch == cfg.SOLVER.OCC_PRED_FROZEN + 1:
            if cfg.MODEL.EXTRA_OCC_BLOCKS != 0:
                frozen_list = ['base.occ_blocks', 'occ_pred']
            else:
                frozen_list = ['base.blocks.0', 'base.blocks.1', 'base.blocks.2', 'occ_pred']
            for idx, para in enumerate(model.named_parameters()):
                for name in frozen_list:
                    if name in para[0]:
                        para[1].requires_grad = False
                        break
        if cfg.MODEL.HEAD_SUP:
            if epoch == 1:
                head_mask = torch.ones(H, dtype=torch.bool, requires_grad=False).to(device)
            head_total_loss = torch.zeros(H, requires_grad=False)
        else:
            head_mask = None
        for n_iter, (img, vid, target_cam, target_view, path) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            if hook_attn_from_model:
                attns_hook = []
            img = img.to(device)
            if len(cfg.MODEL.OCC_TYPES) != 0:
                img_occ, patch_mask = gen_occ(img, vid, cfg, id_img_masks)
                patch_mask = patch_mask.to(device)
            else:
                patch_mask = None
                img_occ = img.clone()

            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                if cfg.MODEL.ZZWEXP:
                    scores, feats, occ_pred, sp_attn, ch_attn = model(img_occ, img, cam_label=target_cam, view_label=target_view, head_suppress=None)
                    score = scores["occ"]
                    if hook_attn_from_model:
                        # for visualize
                        attns_map_occ = rollout(attns_hook[:12], discard_ratio=0.5, head_fusion='mean', occ_token=cfg.MODEL.OCC_AWARE)
                        if cfg.MODEL.TWO_BRANCHED:
                            attns_map_ori = rollout(attns_hook[16:28], discard_ratio=0.5, head_fusion='mean', occ_token=cfg.MODEL.OCC_AWARE)
                        # for training
                        attns_data = hook_out_attention(attns_hook[:12]).to("cuda")
                        attns = {"occ": attns_data}
                    else:
                        attns = None

                    if cfg.MODEL.SAMPLE_HEAD_SUP:
                        update_sample_head_salience(sample_head_salience, path, ch_attn["occ"].to("cpu"))
                    # loss, l_id, l_tri, l_occ, l_ifrc = loss_fn(scores, feats, target, occ_pred=occ_pred, patch_mask=patch_mask, attns=attns)
                    loss, l_id, l_tri, l_occ, l_ifrc, l_tri_div = \
                        loss_fn(scores, feats, target, occ_pred=occ_pred, patch_mask=patch_mask, head_suppress=head_mask, attns=None)

                else:  # origin transreid
                    score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                    if hook_attn_from_model:
                        attns_map_ori = rollout(attns_hook[:12], discard_ratio=0.5, head_fusion='mean')
                    loss, l_id, l_tri = loss_fn(score, feat, target, target_cam)  # target_cam is not used

            scaler.scale(loss).backward()
            '''
            for idx, para in enumerate(model.named_parameters()):
                if para[1].grad == None and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
                    print(para[0])'''
            scaler.step(optimizer)
            scaler.update()

            # compute loss_div sum
            if cfg.MODEL.HEAD_SUP:
                head_total_loss += l_tri_div.cpu()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            if cfg.MODEL.ZZWEXP and not cfg.MODEL.OCC_AUG:
                if cfg.MODEL.OCC_AWARE:
                    loss_occ_meter.update(l_occ.item(), img.shape[0])
                if cfg.MODEL.IFRC:
                    loss_ifrc_meter.update(l_ifrc.item(), img.shape[0])

            loss_id_meter.update(l_id.item(), img.shape[0])
            loss_tri_meter.update(l_tri.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()

            # save images

            if log_json and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0) and n_iter < iter_save_log:
                if cfg.MODEL.ZZWEXP:
                    if epoch == 1 and n_iter == 0:
                        json_logger.save_images(img, 1, '-o')
                        json_logger.save_images(img_occ, 1, '-m')
                        occ_label = visualize_occ_label(patch_mask)
                        json_logger.save_images(occ_label, 1, '-gt')
                    elif epoch % 10 == 0:
                        json_logger.save_images(img, epoch, '-o', id_plus=img.shape[0]*n_iter)
                        json_logger.save_images(img_occ, epoch, '-m', id_plus=img_occ.shape[0]*n_iter)
                        if hook_attn_from_model:
                            json_logger.visualize_attn(img, attns_map_ori, epoch, '-o', prefix='img', id_plus=img.shape[0]*n_iter)
                            if cfg.MODEL.TWO_BRANCHED:
                                json_logger.visualize_attn(img_occ, attns_map_occ, epoch, '-m', prefix='img', id_plus=img_occ.shape[0]*n_iter)
                        if cfg.MODEL.OCC_AWARE:
                            occ_map = visualize_occ_pred(occ_pred)
                            json_logger.save_images(occ_map, epoch, '-occ', id_plus=img.shape[0]*n_iter)

                else: # baseline
                    if epoch == 1:
                        json_logger.save_images(img, 1, '-o')
                    elif epoch % 10 == 0:
                        if hook_attn_from_model:
                            json_logger.visualize_attn(img, attns_map_ori, epoch, '-o', prefix='img')

            if (n_iter + 1) % log_period == 0 and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        if cfg.MODEL.HEAD_SUP:
            # get most salient head id
            least_loss_head = torch.argsort(head_total_loss)[:3]
            if epoch > 40:
                head_mask[least_loss_head] = False

        # append epoch state to json logger

        if log_json and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
            json_logger.append_state({"Epoch": epoch,
                                      "Accuracy": float(acc_meter.avg),
                                      "Learning Rate": scheduler._get_lr(epoch)[0],
                                      "Loss_Total": loss_meter.avg,
                                      "Loss_ID": loss_id_meter.avg,
                                      "Loss_Triplet": loss_tri_meter.avg}, new_epoch=True)
            if cfg.MODEL.ZZWEXP:
                if cfg.MODEL.OCC_AWARE:
                    json_logger.append_state({"Loss_Occ": loss_occ_meter.avg})
                if cfg.MODEL.IFRC:
                    json_logger.append_state({"Loss_Inference": loss_ifrc_meter.avg})

            json_logger.dump_log()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # save checkpoint

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # evaluation

        if epoch % eval_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        attns_hook = []
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        if cfg.MODEL.ZZWEXP:
                            feat, occ_pred, _ = model(img, cam_label=camids, view_label=target_view)
                        else:
                            feat = model(img, cam_label=camids, view_label=target_view)
                        if hook_attn_from_model:
                             attns_map = rollout(attns_hook[:12], discard_ratio=0.5, head_fusion='mean', occ_token=cfg.MODEL.OCC_AWARE)
                        evaluator.update((feat, vid, camid))
                        # save images
                        if log_json and n_iter < iter_save_log:
                            json_logger.save_images(img, epoch, '-o', prefix='query', id_plus=img.shape[0]*n_iter)
                            if hook_attn_from_model:
                                json_logger.visualize_attn(img, attns_map, epoch, '-o', prefix='query', id_plus=img.shape[0]*n_iter)
                            if (cfg.MODEL.ZZWTRY or cfg.MODEL.ZZWEXP) and cfg.MODEL.OCC_AWARE:
                                occ_map = visualize_occ_pred(occ_pred)
                                json_logger.save_images(occ_map, epoch, '-occ', prefix='query', id_plus=img.shape[0]*n_iter)

                cmc, mAP, _, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
            '''
            # TODO: combine reduplicative code
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat,_ = model(img, cam_label=camids, view_label=target_view)

                            if cfg.MODEL.OCCDECODER and cfg.MODEL.USE_DECODER_FEAT == 'glb':
                                evaluator.update((feat[:, :768], vid, camid))
                            else:
                                evaluator.update((feat, vid, camid))
                            # evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        # TODO: check it
                        if cfg.MODEL.OCCDECODER and cfg.MODEL.USE_DECODER_FEAT == 'glb':
                            evaluator.update((feat[:, :768], vid, camid))
                        else:
                            evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()'''
            if log_json and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
                max_rank1 = max(max_rank1, float(cmc[0]))
                if epoch == epochs:  # last epoch
                    json_logger.append_state({"MAX_Rank-1": max_rank1})
                json_logger.append_state({"Rank-1": float(cmc[0]), "Rank-5": float(cmc[4]), "Rank-10": float(cmc[9]), \
                                          "mAP": float(mAP)}, dump=True)



def do_inference(cfg,
                 model,
                 val_loader,
                 val_dataset,
                 num_query):

    device = "cuda"
    log_json = False
    attn_hook = False
    logger = logging.getLogger("transreid.train")
    if log_json and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
        json_logger = JsonLogger(cfg=cfg)
    logger.info("Enter inferencing")
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    attns = []

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    # append hook
    modules = {}
    def get_attn(module, input, output):
        attns.append(output.cpu())
    for name, module in model.named_modules():
        if attn_hook and 'attn_drop' in name and not 'occ_block' in name:
            module.register_forward_hook(get_attn)
            modules[name] = module

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        attns = []
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat, occ_pred = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            if log_json and n_iter == 0:
                json_logger.save_images(img, 0, '-o', prefix='query')
                if attn_hook:
                    attn_mask = rollout(attns[:12], discard_ratio=0.5, head_fusion='mean')
                    json_logger.visualize_attn(img, attn_mask, 0, '-o', prefix='query')
                if cfg.MODEL.OCC_AWARE:
                    occ_map = visualize_occ_pred(occ_pred)
                    json_logger.save_images(occ_map, 0, '-occ', prefix='query')
            if cfg.DATASETS.NAMES == 'occ_reid':
                imgpath_ = [p[:3] + "/" + p for p in imgpath]
                img_path_list.extend(imgpath_)
            else:
                img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _, idx5 = evaluator.compute()
    evaluator.save_npy("no_inf")
    # evaluator.draw_tsne_fig(8, 20, "./figs/tsne.jpg")
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    if cfg.DATASETS.NAMES == 'occ_duke':
        ds = DataStat(set="query")
        ds.summarize(idx5, img_path_list)

    assert idx5.shape[0] == num_query, "test error"
    if log_json:
        json_logger.append_state({"Rank-1": float(cmc[0]), "Rank-5": float(cmc[4]), "Rank-10": float(cmc[9]), \
                                  "mAP": float(mAP)}, dump=True, new_epoch=True)
        json_logger.visualize_matches(idx5, val_dataset, img_path_list)
    return cmc[0], cmc[4]


