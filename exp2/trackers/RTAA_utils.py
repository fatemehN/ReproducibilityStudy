
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import cv2

import torchvision.transforms.functional as tvisf


def _convert_score(score):
    score_temp = score.permute(1, 2, 0).contiguous().view(2, -1) #
    score1 = torch.transpose(score_temp, 0, 1)
    score = F.softmax(score_temp.permute(1, 0), dim=1).data[:, 0].cpu().numpy()
    return score_temp, score1 , score

def _convert_bbox(delta):
    delta1 = delta.permute(1, 2, 0).contiguous().view(4, -1)
    delta = delta1.data.cpu().numpy()
    return delta, delta1


def _bbox_clip(cx, cy, width, height, boundary):
    cx = max(0, min(cx, boundary[1]))
    cy = max(0, min(cy, boundary[0]))
    width = max(10, min(width, boundary[1]))
    height = max(10, min(height, boundary[0]))
    return cx, cy, width, height


def _create_outputs(outputs, size, s_x, penalty_k, window, window_penalty):
    _, _, score = _convert_score(outputs['pred_logits'])
    pred_bbox, delta = _convert_bbox(outputs['pred_boxes'])

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    # scale penalty
    s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                    (sz(size[0]/s_x, size[1]/s_x)))

    # aspect ratio penalty
    r_c = change((size[0]/size[1]) /
                    (pred_bbox[2, :]/pred_bbox[3, :]))
    penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
    pscore = penalty * score

    # window penalty
    pscore = pscore * (1 - window_penalty) + \
                window * window_penalty

    return pscore




def rtaa_attack(net, x_init, x, gt, target_pos, s_x, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    #Normalize for TransT
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inplace = False
    x_adv = Variable(x_init.data, requires_grad=True)
    alpha = eps * 1.0 / iteration

    for i in range(iteration):
        #Normalize for TransT
        x_adv = Variable(x_adv.data, requires_grad=True)
        x_adv.data = x_adv.data.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_adv.data[0] = tvisf.normalize(x_adv.data[0], mean, std, inplace)

        mask_flag = True
        out = net.track_seg(x_adv, mask=mask_flag)
        delta, delta1 = _convert_bbox(out['pred_boxes'])
        _, score1, _ = _convert_score(out['pred_logits'])
        #pscore = _create_outputs(out, target_sz, s_x, penalty_k, window, window_penalty)

        # calculate proposals
        gt_cen = gt 
        gt_cen = np.tile(gt_cen, (score1.shape[0], 1))
        gt_cen[:, 0] = (gt_cen[:, 0] - target_pos[0])*s_x  - s_x / 2
        gt_cen[:, 1] = (gt_cen[:, 1] - target_pos[1])*s_x  - s_x / 2
        gt_cen[:, 2] = gt_cen[:, 2] * s_x
        gt_cen[:, 3] = gt_cen[:, 3] * s_x

        # create pseudo proposals randomly
        gt_cen_pseudo = gt #rect_2_cxy_wh(gt)
        gt_cen_pseudo = np.tile(gt_cen_pseudo, (score1.shape[0], 1))

        rate_xy1 = np.random.uniform(0.3, 0.5)
        rate_xy2 = np.random.uniform(0.3, 0.5)
        rate_wd = np.random.uniform(0.7, 0.9)

        gt_cen_pseudo[:, 0] = (gt_cen_pseudo[:, 0] - target_pos[0] - rate_xy1 * gt_cen_pseudo[:, 2])*s_x - s_x/2
        gt_cen_pseudo[:, 1] = (gt_cen_pseudo[:, 1] - target_pos[1] - rate_xy2 * gt_cen_pseudo[:, 3])*s_x - s_x/2
        gt_cen_pseudo[:, 2] = gt_cen_pseudo[:, 2] * rate_wd * s_x
        gt_cen_pseudo[:, 3] = gt_cen_pseudo[:, 3] * rate_wd * s_x

        delta[0, :] = delta[0, :]*s_x + target_pos[0] - s_x/2
        delta[1, :] = delta[1, :]*s_x + target_pos[1] - s_x/2

        location = np.array([delta[0] - delta[2] / 2, delta[1] - delta[3] / 2, delta[2], delta[3]])
        gt_ce = np.array([gt[0] - gt[2]/2, gt[1] - gt[3]/2, gt[2], gt[3]])

        label = overlap_ratio(location,gt_ce)
   
        # set thresold to define positive and negative samples, following the training step
        iou_hi = 0.6
        iou_low = 0.3 

        # make labels
        y_pos = np.where(label > iou_hi, 1, 0)
        y_pos = torch.from_numpy(y_pos).cuda().long()
        y_neg = np.where(label < iou_low, 0, 1)
        y_neg = torch.from_numpy(y_neg).cuda().long()
        pos_index = np.where(y_pos.cpu() == 1)
        neg_index = np.where(y_neg.cpu() == 0)
        index = np.concatenate((pos_index, neg_index), axis=1)

        # make pseudo lables
        y_pos_pseudo = np.where(label > iou_hi, 0, 1)
        y_pos_pseudo = torch.from_numpy(y_pos_pseudo).cuda().long()
        y_neg_pseudo = np.where(label < iou_low, 1, 0)
        y_neg_pseudo = torch.from_numpy(y_neg_pseudo).cuda().long()

        y_truth = y_pos
        y_pseudo = y_pos_pseudo

        # calculate classification loss
        loss_truth_cls = -F.cross_entropy(score1[index], y_truth[index])
        loss_pseudo_cls = -F.cross_entropy(score1[index], y_pseudo[index])
        loss_cls = (loss_truth_cls - loss_pseudo_cls) * (1)

        # calculate regression loss
        loss_truth_reg = -rpn_smoothL1(delta1, gt_cen, y_pos)
        loss_pseudo_reg = -rpn_smoothL1(delta1, gt_cen_pseudo, y_pos)
        loss_reg = (loss_truth_reg - loss_pseudo_reg) * (5)

        # final adversarial loss
        loss = loss_cls + loss_reg

        
        # calculate the derivative
        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_adv.grad > 0) | (x_adv.grad < 0), x_adv.grad, 0)
        adv_grad = torch.sign(adv_grad)
        #x_adv = x_adv - alpha * adv_grad ##Original line
        x_init =  x_init - alpha * adv_grad

        x_adv = where(x_init > x + eps, x + eps, x_init)
        x_adv = where(x_init < x - eps, x - eps, x_init)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv.data


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    #print('rect1, rect2', len(rect1), len(rect2))
    rect1 = np.transpose(rect1)

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def rpn_smoothL1(input, target, label):
    r"""
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    """
    input = torch.transpose(input, 0, 1)
    pos_index = np.where(label.cpu() == 1)#changed
    target = torch.from_numpy(target).cuda().float()
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], reduction='sum')
    return loss

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]])


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def torch_to_img(img):
    img = to_numpy(torch.squeeze(img, 0))
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original

    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def get_axis_aligned_bbox(region):
    try:
        region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                           region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
    except:
        region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h




