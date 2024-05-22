from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pyexpat import model
from re import search

import cv2
import torch
import vot
import sys

import time
import os
import numpy as np
# from lib.test.tracker.mixformer_online import MixFormerOnline
from lib.test.tracker.mixformer_vit_online import MixFormerOnline
from lib.train.data.processing_utils import transform_image_to_crop
from pytracking.ARcm_seg import ARcm_seg
from pytracking.vot20_utils import *

# import lib.test.parameter.mixformer_online as vot_params
import lib.test.parameter.mixformer_vit_online as vot_params
from vot import Rectangle
import IoU_utils as uts

def iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = (boxes1[2:] - boxes1[:2]).prod() # 1
    area2 = (boxes2[2:] - boxes2[:2]).prod()

    lt = np.maximum(boxes1[:2], boxes2[:2])  # (2)
    rb = np.minimum(boxes1[2:], boxes2[2:])  # (2)

    wh = (rb - lt).clip(min=0)  # (2)
    inter = wh[0] * wh[1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou

def rect_from_mask_local(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return np.array([x0, y0, x1, y1])

class MIXFORMER_ALPHA_SEG(object):
    def __init__(self, tracker,
                 refine_model_name='ARcm_coco_seg', threshold=0.6):
        self.THRES = threshold
        self.tracker = tracker
        '''create tracker'''
        '''Alpha-Refine'''
        project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        refine_path = os.path.join(refine_root, refine_model_name)
        '''2020.4.25 input size: 384x384'''
        # self.alpha = ARcm_seg(refine_path, input_sz=384)
        self.alpha = ARcm_seg(refine_path, input_sz=256)

    def initialize(self, image, bbox):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(bbox).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)
        '''initilize refinement module for specific video'''
        self.alpha.initialize(image, np.array(gt_bbox_np))

    def track(self, img_RGB, bbox, last_perturb):
	###Apply IoU Attack
        im_adv, last_perturb = uts._apply_IoU_attack(self.tracker, img_RGB, bbox, last_perturb)
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(im_adv)
        pred_bbox = outputs['target_bbox'] # x0y0wh of raw
        update_dict = outputs['update_dict']
        # resize_factor = outputs["resize_factor"]
        # prev_state = outputs["prev_state"]
        # olselect_search = outputs["search"]
        # output_sz = outputs["output_sz"]

        '''Step2: Mask report'''
        #mask_arr, bbox_arr = self.alpha.get_mask(img_RGB, np.array(pred_bbox), vis=True)
        #final_mask = (mask_arr > self.THRES).astype(np.uint8)
        # final_mask = (mask_arr > 0.7).astype(np.uint8)
        #self.tracker.update_state_from_refiner(image, bbox_arr, update_dict)

        # if final_mask.astype(np.float).sum() / (pred_bbox[2] * pred_bbox[3]) > 0.1:
        #     mask_bbox = rect_from_mask_local(final_mask) # x0y0x1y1

        #     # x0y0x1y1
        #     pred_bbox_x0y0x1y1 = np.array([pred_bbox[0], pred_bbox[1], pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]])
        #     if iou(pred_bbox_x0y0x1y1, mask_bbox) > 0.7:
        #     # use box from mask branch as fake gt
        #         mask_bbox[2:] = mask_bbox[2:] - mask_bbox[:2] # x0y0x1y1 => x0y0wh 
        #         mask_bbox_torch = torch.from_numpy(mask_bbox)
        #         prev_state_torch = torch.tensor(prev_state)

        #         crop_sz = torch.Tensor([output_sz, output_sz])

        #         new_pred_boxes = transform_image_to_crop(mask_bbox_torch, prev_state_torch, resize_factor, crop_sz)
        #         self.tracker.olselect_update_template(pred_score, new_pred_boxes.cuda(), olselect_search)
        #     else:
        #         self.tracker.olselect_update_template(pred_score, pred_boxes, olselect_search)    
        # else:
        #     # use box from tracker branch as fake gt
        #     self.tracker.olselect_update_template(pred_score, pred_boxes, olselect_search)

        # self.tracker.update_candidates(pred_score, image)

        return pred_bbox, 1


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)



refine_model_name = 'ARcm_coco_seg_only_mask_384' # SEcmnet_ep0250
params = vot_params.parameters("baseline_large_score", model="mixformer_vit_score_imagemae.pth.tar")
mixformer = MixFormerOnline(params, "VOT20")
tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer, refine_model_name=refine_model_name)
handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()

#print(selection)

if not imagefile:
    sys.exit(0)

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
# mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
#mask = make_full_size(selection, (image.shape[1], image.shape[0]))

tracker.H = image.shape[0]
tracker.W = image.shape[1]

tracker.initialize(image, selection)
bbox = selection
last_perturb = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
while True:
	imagefile = handle.frame()
	if not imagefile:
		break
	image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
	bbox, confidence = tracker.track(image, bbox, last_perturb)
	results = Rectangle(*bbox)
	handle.report(results, confidence)
