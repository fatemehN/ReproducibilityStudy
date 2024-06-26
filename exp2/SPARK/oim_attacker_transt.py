# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import random, math
import torch.nn.functional as F
import visdom
from pysot.core.config import cfg
import matplotlib.pyplot as plt
from pysot.models.loss import select_cross_entropy_loss
from pysot.datasets.anchor_target import AnchorTarget
from anchor_target import anchor_Target4TransT
from pysot.utils.bbox import center2corner, Center, get_axis_aligned_bbox
from math import cos, sin, pi
import os
import cv2
import torchvision.transforms.functional as tvisf

class OIMAttacker():
    def __init__(self, type, max_num=10, eplison=1, inta=10, lamb=0.00001, norm_type='L_inf', apts_num=2, reg_type='weighted',accframes=30):

        self.type = type
        # parameters for bim
        self.eplison = eplison
        self.inta = inta
        self.norm_type = norm_type
        self.reg_type = reg_type
        self.max_num = max_num
        self.lamb = lamb
        self.v_id = 0
        # self.st = st()
        self.apts_num = apts_num
        self.target_traj = []
        self.prev_delta = None
        self.tacc = True
        self.meta_path = []
        self.acc_iters = 0
        self.weights = []
        self.weight_eplison = 1
        self.lamb_momen = 1
        self.target_pos = []
        self.accframes = accframes

    def save_tensorasimg(self, path, var):
        np_var = torch.squeeze(var).detach().cpu()
        if len(np_var.shape) == 2:
            np_var = np_var.view(1, np_var.shape[0], np_var.shape[1])
        np_var = np_var.numpy()
        np_var = np.transpose(np_var, (1, 2, 0))
        cv2.imwrite(path, np_var, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def attack(self, tracker, img, prev_perts=None, weights=None, APTS=False, OPTICAL_FLOW=False, ADAPT=False, Enable_same_prev=True):
        """
        args:
            tracker, img(np.ndarray): BGR image
        return:
            adversirial image
        """
        # calculate x crop size
        w_x = tracker.size[0] + (4 - 1) * ((tracker.size[0] + tracker.size[1]) * 0.5)
        h_x = tracker.size[1] + (4 - 1) * ((tracker.size[0] + tracker.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        x_crop = tracker.get_subwindow(img, tracker.center_pos,
                                       tracker.instance_size,
                                       round(s_x), tracker.channel_average)
        #x_crop = img
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        inplace = False

        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop[0] = tvisf.normalize(x_crop[0], mean, std, inplace)


        outputs = tracker.net.track(x_crop)
        # score = tracker._convert_score(outputs['pred_logits'])
        score = F.softmax(outputs['pred_logits'].contiguous()).detach().cpu().numpy() #.view(2, -1).permute(1, 0)
        pred_bbox = tracker._convert_bbox(outputs['pred_boxes'])

        # pred_bbox = outputs['target_bbox']
        # score = outputs['best_score']

        dscore = score[:, :, 1] - score[:, :, 0]
        

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(tracker.size[0]/s_x, tracker.size[1]/s_x)))

        # aspect ratio penalty
        r_c = change((tracker.size[0]/tracker.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * tracker.penalty_k)
        pscore = penalty * dscore

        # window penalty
        diff_cls = pscore * (1 - tracker.window_penalty) + \
                 tracker.window * tracker.window_penalty
        
        # print(cls.shape)
        
        # diff_cls = cls[:, :, 1] - cls[:, :, 0]
        label_cls = torch.from_numpy(diff_cls).ge(0).float()
        

        if self.type == 'UA':
            adv_cls, same_prev= self.ua_label(tracker, s_x, outputs, img.shape[:2])
            adv_cls = adv_cls.long()

        max_iteration = self.max_num
        if cfg.CUDA:
            pert = torch.zeros(x_crop.size()).cuda()
        else:
            pert = torch.zeros(x_crop.size())

        if prev_perts is None or (same_prev == False and Enable_same_prev == True):
            if cfg.CUDA:
                prev_perts = torch.zeros(x_crop.size()).cuda()
                weights = torch.ones(x_crop.size()).cuda()#torch.ones(1).cuda()
            else:
                prev_perts = torch.zeros(x_crop.size())
                weights = torch.ones(x_crop.size())#torch.ones(1)
        else:
            if APTS == False:
                if self.reg_type == 'weighted':
                    pert_sum = torch.mul(weights, prev_perts).sum(0)
                else:
                    pert_sum = prev_perts.sum(0)
                adv_x_crop = x_crop + pert_sum
                adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
                pert_true = adv_x_crop - x_crop

                if cfg.ATTACKER.GEN_ADV_IMG:
                    adv_img = tracker.get_orgimg(img, x_crop, tracker.center_pos,
                                                 cfg.TRACK.INSTANCE_SIZE,
                                                 round(s_x), tracker.channel_average)
                else:
                    adv_img = None

                return x_crop, pert_true, prev_perts, weights,adv_img
            else:
                max_iteration = self.apts_num
                if self.tacc == False:
                    if cfg.CUDA:
                        prev_perts = torch.zeros(x_crop.size()).cuda()
                        weights = torch.ones(x_crop.size()).cuda()  # torch.ones(1).cuda()
                    else:
                        prev_perts = torch.zeros(x_crop.size())
                        weights = torch.ones(x_crop.size())  # torch.ones(1)

        # start attack
        losses = []
        m = 0

        if cfg.CUDA:
            momen = torch.zeros(x_crop.size()).cuda()
        else:
            momen = torch.zeros(x_crop.size())

        self.acc_iters += max_iteration
        while m < max_iteration:
            #if isinstance(tracker.net.zf, list):
            #    zf = torch.cat(tracker.net.zf, 0)
            #else:
            #    zf = tracker.net.zf
            # print('pp', prev_perts)
            data = {
            #    'template_zf': zf.detach(),
                'search': x_crop.detach(),
                'pert': pert.detach(),
                'prev_perts': prev_perts.detach(),
                'label_cls': label_cls.detach(),
                'adv_cls': adv_cls.detach(),
                'weights':weights.detach(),
                'momen':momen.detach()
            }

            data['pert'].requires_grad = True
            #data['weights'].requires_grad = True
            pert, loss, update_cls, momen, weights = self.oim_once(tracker, data)
            losses.append(loss)
            m += 1

        self.opt_flow_prev_xcrop = x_crop
        if cfg.CUDA:
            prev_perts = torch.cat((prev_perts, pert), 0).cuda()
            weights = torch.cat((weights, torch.ones(x_crop.size()).cuda()),0).cuda()
        else:
            prev_perts = torch.cat((prev_perts, pert), 0)
            weights = torch.cat((weights, torch.ones(x_crop.size())), 0)

        if self.reg_type=='weighted':
            pert_sum = torch.mul(weights,prev_perts).sum(0)
        else:
            pert_sum = prev_perts.sum(0)

        adv_x_crop = x_crop + pert_sum
        adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
        pert_true = adv_x_crop - x_crop
        if cfg.ATTACKER.GEN_ADV_IMG:
            adv_img = tracker.get_orgimg(img, x_crop, tracker.center_pos,
                                         cfg.TRACK.INSTANCE_SIZE,
                                        round(s_x), tracker.channel_average)
        else:
            adv_img = None

        if prev_perts.shape[0]>self.accframes:
            prev_perts = prev_perts[-self.accframes:,:,:,:]
            weights = weights[-self.accframes:,:,:,:]

        return adv_x_crop, pert_true, prev_perts, weights, adv_img

    def ua_label(self, tracker, s_x, outputs, img_sh):

        score = tracker._convert_score(outputs['pred_logits'])
        pred_bbox = tracker._convert_bbox(outputs['pred_boxes']) #, tracker.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(tracker.size[0]/s_x, tracker.size[1]/s_x)))

        # aspect ratio penalty
        r_c = change((tracker.size[0]/tracker.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * tracker.penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - tracker.window_penalty) + \
                 tracker.window * tracker.window_penalty
        
        best_idx = np.argmax(pscore)

        obj_bbox = pred_bbox[:, best_idx]

        obj_bbox = obj_bbox * s_x
        cx = obj_bbox[0] + tracker.center_pos[0] - s_x / 2
        cy = obj_bbox[1] + tracker.center_pos[1] - s_x / 2
        width = obj_bbox[2]
        height = obj_bbox[3]
        # print('cx, cy', cx, cy, width, height, img_sh)
        # clip boundary
        cx, cy, width, height = tracker._bbox_clip(cx, cy, width, height, img_sh)

        obj_pos = np.array([cx, cy])
        size = np.array([width, height])
        
        template_size = np.array([size[0] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1]), \
                                  size[1] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1])])
        context_size = template_size * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        same_prev = False
        # validate delta meets the requirement of obj
        if self.prev_delta is not None:
            diff_pos = np.abs(self.prev_delta - obj_pos)
            if (size[0] // 2 < diff_pos[0] and diff_pos[0] < context_size[0] // 2) \
                    and (size[1] // 2 < diff_pos[1] and diff_pos[1] < context_size[1] // 2):
                delta = self.prev_delta
                same_prev = True
            else:
                delta = []
                delta.append(random.choice((1, -1)) * random.randint(size[0] // 2, context_size[0] // 2))
                delta.append(random.choice((1, -1)) * random.randint(size[1] // 2, context_size[1] // 2))
                delta = obj_pos + np.array(delta)
                self.prev_delta = delta
        else:
            delta = []
            delta.append(random.choice((1, -1)) * random.randint(size[0] // 2, context_size[0] // 2 - size[0] // 2))
            delta.append(random.choice((1, -1)) * random.randint(size[1] // 2, context_size[1] // 2 - size[1] // 2))
            delta = obj_pos + np.array(delta)
            self.prev_delta = delta

        desired_pos = context_size / 2 + delta
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(desired_pos[0], downbound_pos[0], upbound_pos[0])
        desired_pos[1] = np.clip(desired_pos[1], downbound_pos[1], upbound_pos[1])
        if cfg.ATTACKER.EVAL:
            desired_pos_abs = tracker.center_pos + delta
            tpos = []
            tpos.append(desired_pos_abs[0])
            tpos.append(desired_pos_abs[1])
            self.target_traj.append(tpos)

        #desired_bbox = self._get_bbox(desired_pos, size)

        cx = desired_pos[0]
        cy = desired_pos[1]
        h = size[0]
        w = size[1]
        desired_bbox = center2corner(Center(cx - w / 2, cy - h / 2, w, h))

        # anchor_target = AnchorTarget()
        w = 1
        # results = anchor_target(desired_bbox, w)
        # print(pred_bbox.shape, 'pred-bbox')
        obj_locs = pred_bbox * s_x
        obj_locs[:, 0] = obj_locs[:, 0] + tracker.center_pos[0] - s_x / 2
        obj_locs[:, 1] = obj_locs[:, 1] + tracker.center_pos[1] - s_x / 2
        obj_locs[:, 2] = obj_locs[:, 2]
        obj_locs[:, 3] = obj_locs[:, 3]

        results = anchor_Target4TransT(desired_bbox, w, obj_locs)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)[0]
        adv_cls = torch.zeros(results[0].shape)
        # print('adv cls size, over lap shaoe', adv_cls.shape, overlap.shape)
        adv_cls[max_pos, ...] = 1
        adv_cls = adv_cls.view(1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2])

        self.target_pos = desired_pos

        return adv_cls,same_prev

    def ta_label(self, tracker, scale_z, outputs):

        b, a2, h, w = outputs['cls'].size()
        center_pos = tracker.center_pos
        size = tracker.size
        desired_pos = np.array(self.target_traj[self.v_id])
        same_prev = False
        if self.v_id > 1:
            prev_desired_pos = np.array(self.target_traj[self.v_id - 1])
            if np.linalg.norm(prev_desired_pos - desired_pos) < 5:
                same_prev = True
        template_size = np.array([size[0] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1]), \
                                  size[1] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1])])
        context_size = template_size * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        delta = desired_pos - center_pos
        desired_pos = delta + context_size / 2
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(desired_pos[0], downbound_pos[0], upbound_pos[0])
        desired_pos[1] = np.clip(desired_pos[1], downbound_pos[1], upbound_pos[1])
        desired_bbox = self._get_bbox(desired_pos, size)

        anchor_target = AnchorTarget()

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)[0]
        print(max_pos.shape, overlap.shape)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        adv_cls = adv_cls.view(1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2])

        self.target_pos = desired_pos

        return adv_cls, same_prev

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def oim_once(self, tracker, data):
        if cfg.CUDA:
        #    zf = data['template_zf'].cuda()
            search = data['search'].cuda()
            pert = data['pert'].cuda()
            prev_perts = data['prev_perts'].cuda()
            adv_cls = data['adv_cls'].cuda()
            weights = data['weights'].cuda()
            momen = data['momen'].cuda()
        else:
        #    zf = data['template_zf']
            search = data['search']
            pert = data['pert']
            prev_perts = data['prev_perts']
            adv_cls = data['adv_cls']
            weights = data['weights']
            momen = data['momen']

        #track_model = tracker.net

        #zf_list = []
        #if zf.shape[0] > 1:
        #    for i in range(0, zf.shape[0]):
        #        zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
        #else:
        #    zf_list = zf

        # get feature
        if self.reg_type == 'weighted':
            pert_sum = torch.mul(weights, prev_perts).sum(0)
        else:
            pert_sum = prev_perts.sum(0)

        #xf = track_model.backbone(
        #    search + pert + pert_sum.view(1, prev_perts.shape[1], prev_perts.shape[2], prev_perts.shape[3]))
        #if cfg.ADJUST.ADJUST:
        #    xf = track_model.neck(xf)
        #cls, loc = track_model.rpn_head(zf_list, xf)
        #print('prev_perts', prev_perts.shape)
        xf = search + pert + pert_sum #.view(1, prev_perts.shape[1], prev_perts.shape[2], prev_perts.shape[3])
        #Normalize it for TransT
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        inplace = False
        xf = xf.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        xf[0] = tvisf.normalize(xf[0], mean, std, inplace)

        outputs = tracker.net.track(xf)
        #cls = tracker.net.log_softmax(outputs['cls'])
        cls = F.log_softmax(outputs['pred_logits'])

        # get loss1
        #cls = track_model.log_softmax(cls)
        #cls = F.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, adv_cls)

        # regularization loss
        if cfg.CUDA:
            c_prev_perts = torch.cat((prev_perts, pert), 0).cuda()
            #c_weights = torch.cat((weights, torch.ones(pert.size()).cuda()), 0).cuda()
        else:
            c_prev_perts = torch.cat((prev_perts, pert), 0)
            #c_weights = torch.cat((weights, torch.ones(pert.size())), 0)

        t_prev_perts = c_prev_perts.view(c_prev_perts.shape[0]*c_prev_perts.shape[1],
                                         c_prev_perts.shape[2] * c_prev_perts.shape[3])
        #t_weights = c_weights.view(c_weights.shape[0],
        #  c_weights.shape[1] * c_weights.shape[2] * c_weights.shape[3])

        if self.reg_type == 'L21':
            reg_loss = torch.norm(t_prev_perts, 2, 1).sum()  # +torch.norm(pert,2)
        elif self.reg_type == 'L2':
            reg_loss = torch.norm(t_prev_perts, 2)
        elif self.reg_type == 'L_inf':
            reg_loss = torch.max(torch.abs(t_prev_perts))
            #print("reg_loss{}".format(reg_loss))
        elif self.reg_type == 'weighted':
            reg_loss = torch.norm(t_prev_perts, 2, 1).sum()
        else:
            reg_loss = 0.

        total_loss = cls_loss + self.lamb * reg_loss
        #print('total loss of SPARK is', total_loss)
        total_loss.backward(retain_graph=True)

        x_grad = -data['pert'].grad

        if self.reg_type == 'weighted':
            pert_sum = torch.mul(weights, prev_perts).sum(0)

        adv_x = search

        if self.norm_type == 'L_inf':

            x_grad = torch.sign(x_grad)
            adv_x = adv_x + pert+ pert_sum + self.eplison * x_grad
            pert = adv_x - search-pert_sum
            pert = torch.clamp(pert, -self.inta, self.inta)

        elif self.norm_type == 'L_1':
            adv_x = adv_x + pert + pert_sum + self.eplison * x_grad
            pert = adv_x - search-pert_sum
            norm = torch.sum(torch.abs(pert))
            pert = torch.min(pert*self.inta/norm, pert)

        elif self.norm_type == 'L_2':

            x_grad = x_grad / torch.norm(x_grad)
            adv_x = adv_x + pert + pert_sum  + self.eplison * x_grad
            pert = adv_x - search-pert_sum
            pert = torch.clamp(pert / torch.norm(pert), -self.inta, self.inta)

        elif self.norm_type=='Momen':

            momen = self.lamb_momen*momen+x_grad/torch.norm(x_grad,1)
            adv_x = adv_x + pert + pert_sum + self.eplison*torch.sign(momen)
            pert = adv_x-search-pert_sum
            pert = torch.clamp(pert,-self.inta,self.inta)

        p_search = search + pert + pert_sum
        p_search = torch.clamp(p_search, 0, 255)
        pert = p_search - search - prev_perts.sum(0)

        return pert, total_loss, cls, momen, weights

    def target_traj_gen(self, init_rect, vid_h, vid_w, vid_l):

        target_traj = []
        pos = []
        pos.append(init_rect[0] + init_rect[2] / 2)
        pos.append(init_rect[1] + init_rect[3] / 2)
        target_traj.append(pos)
        for i in range(0, vid_l - 1):
            tpos = []
            if i % 50 == 0:
                deltay = random.randint(-10, 10)
                deltax = random.randint(-10, 10)
            elif i % 10 == 0:
                deltay = random.randint(0, 1)
                deltax = random.randint(0, 1)
            elif i % 5 == 0:
                deltay = random.randint(-1, 0)
                deltax = random.randint(-1, 0)
            tpos.append(np.clip(target_traj[i][0] + deltax, 0, vid_w - 1))
            tpos.append(np.clip(target_traj[i][1] + deltay, 0, vid_h - 1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj

    def target_traj_gen_supervised(self, init_rect, vid_h, vid_w, vid_l, gt_traj):

        target_traj = []
        w, h = gt_traj[0][2], gt_traj[0][3]
        w_z = w + cfg.TRACK.CONTEXT_AMOUNT * np.sum(np.array([w, h]))
        h_z = h + cfg.TRACK.CONTEXT_AMOUNT * np.sum(np.array([w, h]))
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        deltax, deltay = -s_x / 5, -s_x / 5
        for i in range(vid_l):
            pos = []
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_traj[i]))
            pos.append(cx + deltax)
            pos.append(cy + deltay)
            target_traj.append(pos)
        self.target_traj = target_traj
        return target_traj

    def target_traj_gen_custom(self, init_rect, vid_h, vid_w, vid_l, type=1):
        '''
        type 1 : ellipse
        type 2 : rectangle
        type 3 : triangle
        '''
        target_traj = []
        initpos = np.array([init_rect[0] - init_rect[2] / 2, init_rect[1] - init_rect[3] / 2])
        target_traj.append(initpos)

        # initial start_point , shape of traj
        def ellipse(t, a):
            return a * t * cos(t), a * t * sin(t)

        def rectangle(t, a):
            if t < 0.5:
                return t * a, 0
            elif t >= 0.5 and t < 1:
                return 0.5 * a, -(t - 0.5) * a
            elif t >= 1 and t < 2:
                return -(t - 1) * a + 0.5 * a, -0.5 * a
            elif t >= 2 and t < 3:
                return -0.5 * a, (t - 2) * a - 0.5 * a
            else:
                return (t - 3) * a - 0.5 * a, 0.5 * a

            return 0, 0

        def triangle(t, r):
            if t < 1:
                return 0.5 * r - r * t * cos(pi / 3), -r * t * sin(pi / 3),
            elif t < 2:
                return -(t - 1) * r * cos(pi / 3), (t - 2) * r * sin(pi / 3)
            else:
                return (t - 2.5) * r, 0

        def line(t, r, theta):
            """line genetate line traj

            Arguments:
                t {float} -- time
                r {float} -- length
                theta {float} -- angle

            Returns:
                [float,float] -- position
            """
            return t * r * cos(theta), t * r * sin(theta)

        r = 2 * min(vid_w, vid_h) / 2

        for i in range(0, vid_l - 1):
            tpos = []
            if type == 1:
                t = 6 * pi * i / vid_l
                x, y = ellipse(t, r / (pi * 8))
            if type == 2:
                t = 4. * i / vid_l
                x, y = rectangle(t, r / 2)

            if type == 3:
                t = 3. * i / vid_l
                x, y = triangle(t, r / 2)

            if type == 4:
                t = 1. * i / vid_l
                x, y = line(t, r, -pi * 0.4)

            tpos.append(np.clip(x + initpos[0], 0, vid_w - 1))
            tpos.append(np.clip(y + initpos[1], 0, vid_h - 1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj

    def _get_bbox(self, center_pos, shape):
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = center_pos * scale_z
        bbox = center2corner(Center(cx - w / 2, cy - h / 2, w, h))
        return bbox