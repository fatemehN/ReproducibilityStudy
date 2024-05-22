import torch
#from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import numpy as np
#from common_path import train_set_path_ as dataset_dir

#import imutils


def img2tensor(img_arr):
    '''float64 ndarray (H,W,3) ---> float32 torch tensor (1,3,H,W)'''
    img_arr = img_arr.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1) # channel first
    img_arr = img_arr[np.newaxis, :, :, :]
    init_tensor = torch.from_numpy(img_arr)  # (1,3,H,W)
    return init_tensor
def normalize(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor
def tensor2img(tensor):
    '''(0,255) tensor ---> (0,255) img'''
    '''(1,3,H,W) ---> (H,W,3)'''
    tensor = tensor.squeeze(0).permute(1,2,0)
    img = tensor.cpu().numpy().clip(0,255).astype(np.uint8) 
    return img

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image
		
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0], :])