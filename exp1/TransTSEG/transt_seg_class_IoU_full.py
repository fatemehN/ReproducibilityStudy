from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time
'''Refine module & Pytracking base trackers'''
# from common_path import *
import os
'''2020.4.24 Use new pytracking library(New DiMP)'''
from pytracking.evaluation import Tracker
'''2020.4.15 ARcm_seg model'''
'''other utils'''
from pytracking.vot20_utils import *
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
import pytracking.evaluation.vot2020 as vot
import IoU_full_utils as Iuts
''''''
class TRANST_SEG(object):

    def __init__(self, name, net, window_penalty=0.59, penalty_k=0.265, exemplar_size=128, instance_size=256, mask_threshold = 0.5):
        self.name = name
        self.net = net
        self.window_penalty = window_penalty
        self.penalty_k = penalty_k
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.mask_threshold = mask_threshold

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def _convert_mask(self, delta):

        delta = delta.squeeze(0).squeeze(0)
        delta = delta.data.cpu().numpy()

        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def map_mask_back(self, im, center_pos, instance_size, s_x, mask, mode=cv2.BORDER_REPLICATE):
        """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        H, W = (im.shape[0], im.shape[1])
        base = np.zeros((H, W))
        x_center, y_center = center_pos.tolist()

        # Crop image

        if s_x < 1 or s_x < 1:
            raise Exception('Too small bounding box.')
        c = (s_x + 1) / 2

        x1 = int(np.floor(x_center - c + 0.5))
        x2 = int(x1 + s_x - 1)

        y1 = int(np.floor(y_center - c + 0.5))
        y2 = int(y1 + s_x -1)

        x1_pad = int(max(0., -x1))
        y1_pad = int(max(0., -y1))
        x2_pad = int(max(0., x2 - W + 1))
        y2_pad = int(max(0., y2 - H + 1))

        '''pad base'''
        base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
        '''Resize mask'''
        mask_rsz = cv2.resize(mask, (s_x, s_x))
        '''fill region with mask'''
        base_padded[y1 + y1_pad:y2 + y1_pad + 1, x1 + x1_pad:x2 + x1_pad + 1] = mask_rsz.copy()
        '''crop base_padded to get final mask'''
        final_mask = base_padded[y1_pad:y1_pad + H, x1_pad:x1_pad + W]
        assert (final_mask.shape == (H, W))
        return final_mask

    def constraint_mask(self, mask, bbox):
        """
        mask: shape (H, W)
        bbox: list [x1, y1, w, h]
        """
        x1 = np.int(np.floor(bbox[0]))
        y1 = np.int(np.floor(bbox[1]))
        x2 = np.int(np.ceil(bbox[0]+bbox[2]))
        y2 = np.int(np.ceil(bbox[1]+bbox[3]))
        mask[0:y1+1,:] = 0
        mask[y2:,:] = 0
        mask[:,0:x1+1] = 0
        mask[:,x2:] = 0
        if mask.max() == 0:
            yp1 = np.int(np.floor(bbox[1]+bbox[3]/4))
            yp2 = np.int(np.ceil(bbox[1]+3*bbox[3]/4))
            xp1 = np.int(np.floor(bbox[0]+bbox[2]/4))
            xp2 = np.int(np.ceil(bbox[0]+3*bbox[2]/4))
            mask[yp1:yp2,xp1:xp2] = 1
        return mask

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, mask):
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()

        region = rect_from_mask(mask)
        gt_bbox_np = np.array(region).astype(np.float32)
        bbox = gt_bbox_np

        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
        
        # initialize template feature
        self.net.template(z_crop)


    def track(self, image):
        # calculate x crop size
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        # get crop
        x_crop_ori = self.get_subwindow(image, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)


        ###Apply Adverserail attack
        # x_crop = x_crop_ori.squeeze(0).permute(1, 2, 0)
        # img_adv, last_preturb = Iuts._apply_IoU_attack(self.net, x_crop, pred_bbox, last_preturb, self.size, 
        #                                               s_x, image.shape[:2], self.center_pos, self.window)
        
        # im_adv = torch.from_numpy(img_adv).unsqueeze(0).permute(0, 3, 1, 2)

        # normalize
        x = x_crop_ori.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x[0] = tvisf.normalize(x[0], self.mean, self.std, self.inplace)

        # track
        mask_flag = True
        outputs = self.net.track_seg(x.cuda(), mask=mask_flag)
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]/s_x, self.size[1]/s_x)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if mask_flag == True:
            pred_mask_search = self._convert_mask(outputs['pred_masks'])
            pred_mask = self.map_mask_back(image, self.center_pos, self.instance_size, s_x, pred_mask_search,
                                      mode=cv2.BORDER_CONSTANT)
            final_mask = (pred_mask > self.mask_threshold).astype(np.uint8)
            # final_mask = self.constraint_mask(final_mask, bbox)
            mask_search = (pred_mask_search > self.mask_threshold).astype(np.uint8)

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        x_crop_return = x_crop_ori.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # out = {'target_bbox': bbox,
        #        'target_mask': final_mask,
        #        'best_score': pscore[best_idx]}
        return bbox, final_mask

def run_vot_exp(name, window, penalty_k, mask_threshold, net_path, save_root, VIS=False):

    # torch.set_num_threads(1)
    # torch.cuda.set_device(CUDA_ID)  # set GPU id
    if VIS and (not os.path.exists(save_root)):
        os.mkdir(save_root)

    net = NetWithBackbone(net_path=net_path, use_gpu=True)
    tracker = TRANST_SEG(name=name, net=net,
                         window_penalty=window, penalty_k=penalty_k,
                         mask_threshold=mask_threshold,
                         exemplar_size=128, instance_size=256)
    handle = vot.VOT("mask")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB) # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    mask = make_full_size(selection, (image.shape[1], image.shape[0]))

    last_preturb = np.zeros(image.shape)
    pred_bbox = rect_from_mask(mask)
    tracker.initialize(image, mask)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        img_adv, last_preturb = Iuts._apply_IoU_attack(tracker, image, pred_bbox, last_preturb)

        pred_bbox, m = tracker.track(img_adv)
        handle.report(m)
        