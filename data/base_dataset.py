import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, shape):
    h,w,nc = shape
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=transforms.InterpolationMode.BICUBIC, normalize=True):
    osize = (opt.loadSize, opt.loadSize)
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: cv2.resize(img, osize)))
    transform_list += [transforms.ToTensor()]

    # if opt.isTrain and not opt.no_flip:
    #     transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
