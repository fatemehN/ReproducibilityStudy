##IoU Auxiliary functions 

import numpy as np


def _apply_IoU_attack(tracker, img, bbox, last_preturb):
    # black-box IoU attack
    image = img
    last_gt = bbox
    perturb_max = 10000
    # heavy noise image
    heavy_noise = np.random.randint(-1, 2, (image.shape[0], image.shape[1], image.shape[2])) * 128
    image_noise = image + heavy_noise
    image_noise = np.clip(image_noise, 0, 255)

    noise_sample = image_noise - 128
    clean_sample_init = image.astype(np.float) - 128
    image_noise = image_noise.astype(np.uint8)
    # query
    outputs_orig = tracker.track(image)
    #print('output tracker', outputs_orig, outputs_orig[1]['target_bbox'])
    pred_orig = outputs_orig['target_bbox']
    outputs_target = tracker.track(image_noise)
    pred_target = outputs_target['target_bbox']
    target_score = overlap_ratio(np.array(pred_orig), np.array(pred_target))
    adversarial_sample = image.astype(np.float) - 128

    if target_score < 0.8:
        # parameters
        n_steps = 0
        epsilon = 0.05
        delta = 0.05
        weight = 0.5
        para_rate = 0.9
        # Move a small step
        while True:
            # Initialize with previous perturbations
            clean_sample = clean_sample_init + weight * last_preturb
            trial_sample = clean_sample + forward_perturbation(
                epsilon * get_diff(clean_sample, noise_sample), adversarial_sample, noise_sample)
            trial_sample = np.clip(trial_sample, -128, 127)
            outputs_adv = tracker.track((trial_sample + 128).astype(np.uint8))
            pred_adv = outputs_adv['target_bbox']
            # IoU score
            threshold_1 = overlap_ratio(np.array(pred_orig), np.array(pred_adv))
            threshold_2 = overlap_ratio(np.array(last_gt), np.array(pred_adv))
            threshold = para_rate * threshold_1 + (1 - para_rate) * threshold_2
            adversarial_sample = trial_sample
            break

        while True:
            # Tangential direction
            d_step = 0
            while True:
                d_step += 1
                # print("\t#{}".format(d_step))
                trial_samples = []
                score_sum = []
                for i in np.arange(10):
                    trial_sample = adversarial_sample + orthogonal_perturbation(delta,
                                                                                adversarial_sample,
                                                                                noise_sample)
                    trial_sample = np.clip(trial_sample, -128, 127)
                    # query
                    outputs_adv = tracker.track((trial_sample + 128).astype(np.uint8))
                    pred_adv = outputs_adv['target_bbox']
                    # IoU score
                    score_1 = overlap_ratio(np.array(pred_orig),
                                            np.array(pred_adv))
                    score_2 = overlap_ratio(np.array(last_gt), np.array(pred_adv))
                    score = para_rate * score_1 + (1 - para_rate) * score_2
                    score_sum = np.hstack((score_sum, score))
                    trial_samples.append(trial_sample)
                d_score = np.mean(score_sum <= threshold)
                if d_score > 0.0:
                    if d_score < 0.3:
                        delta /= 0.9
                    elif d_score > 0.7:
                        delta *= 0.9
                    adversarial_sample = np.array(trial_samples)[np.argmin(np.array(score_sum))]
                    threshold = score_sum[np.argmin(np.array(score_sum))]
                    break
                elif d_step >= 5 or delta > 0.3:
                    break
                else:
                    delta /= 0.9
            # Normal direction
            e_step = 0
            while True:
                trial_sample = adversarial_sample + forward_perturbation(
                    epsilon * get_diff(adversarial_sample, noise_sample), adversarial_sample,
                    noise_sample)
                trial_sample = np.clip(trial_sample, -128, 127)
                # query
                outputs_adv = tracker.track((trial_sample + 128).astype(np.uint8))
                pred_adv = outputs_adv['target_bbox']
                l2_norm = np.mean(get_diff(clean_sample_init, trial_sample))
                # IoU score
                threshold_1 = overlap_ratio(np.array(pred_orig), np.array(pred_adv))
                threshold_2 = overlap_ratio(np.array(last_gt), np.array(pred_adv))
                threshold_sum = para_rate * threshold_1 + (1 - para_rate) * threshold_2

                if threshold_sum <= threshold:
                    adversarial_sample = trial_sample
                    epsilon *= 0.9
                    threshold = threshold_sum
                    break
                elif e_step >= 30 or l2_norm > perturb_max:
                    break
                else:
                    epsilon /= 0.9
            n_steps += 1

            if threshold <= target_score or l2_norm > perturb_max:
                adversarial_sample = np.clip(adversarial_sample, -128, 127)
                break

        last_preturb = adversarial_sample - clean_sample
        img = (adversarial_sample + 128).astype(np.uint8)
    else:
        image = img
        adversarial_sample = image + last_preturb
        adversarial_sample = np.clip(adversarial_sample, 0, 255)
        img = adversarial_sample.astype(np.uint8)
    return img, last_preturb

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
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

def orthogonal_perturbation(delta, prev_sample, target_sample):
    size = int(max(prev_sample.shape[0]/4, prev_sample.shape[1]/4, 224))
    prev_sample_temp = np.resize(prev_sample, (size, size, 3))
    target_sample_temp = np.resize(target_sample, (size, size, 3))
    # Generate perturbation
    perturb = np.random.randn(size, size, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample_temp, prev_sample_temp))
    # Project perturbation onto sphere around target
    diff = (target_sample_temp - prev_sample_temp).astype(np.float32)
    diff /= get_diff(target_sample_temp, prev_sample_temp)
    diff = diff.reshape(3, size, size)
    perturb = perturb.reshape(3, size, size)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    perturb = perturb.reshape(size, size, 3)
    perturb_temp = np.resize(perturb, (prev_sample.shape[0], prev_sample.shape[1], 3))
    return perturb_temp

def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb

def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, sample_1.shape[0], sample_1.shape[1])
    sample_2 = sample_2.reshape(3, sample_2.shape[0], sample_2.shape[1])
    sample_1 = np.resize(sample_1, (3, 271, 271))
    sample_2 = np.resize(sample_2, (3, 271, 271))

    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)
