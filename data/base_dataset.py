import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from data.face_color_transfer import face_color_transfer
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

def get_params(opt):
    new_h = new_w = opt.loadSize            

    res = {}
    res['osize'] = (new_h, new_w)
    if opt.isTrain and opt.random_flip and (random.random() > 0.5):
        res['vflip'] = True
    else:
        res['vflip'] = False

    if opt.isTrain and opt.random_resized_crop and (np.random.random() < 0.8):
        lx = opt.fineSize + random.randint(0, np.maximum(0, new_w - opt.fineSize))
        ly = opt.fineSize + random.randint(0, np.maximum(0, new_h - opt.fineSize))
        sx = random.randint(0, np.maximum(0, new_w - lx))
        sy = random.randint(0, np.maximum(0, new_h - ly))
        res['resized_crop'] = (sx, sy, sx + lx, sy + ly)
    else:
        res['resized_crop'] = False
    
    if opt.isTrain and opt.random_padding_crop and (np.random.random() > 0.8):
        lx = opt.fineSize + random.randint(0, np.maximum(0, new_w - opt.fineSize))
        ly = opt.fineSize + random.randint(0, np.maximum(0, new_h - opt.fineSize))
        sx = random.randint(0, np.maximum(0, new_w - lx))
        sy = random.randint(0, np.maximum(0, new_h - ly))
        x_len = opt.loadSize - lx
        y_len = opt.loadSize - ly
        top = random.randint(0, x_len)
        left = random.randint(0, y_len)
        bottom = x_len - top
        right = y_len - left
        res['padding_crop'] = (sx, sy, sx + lx, sy + ly, top, bottom, left, right)
    else:
        res['padding_crop'] = False
    
    if opt.isTrain and opt.random_illuminant_adjust and (np.random.random() > 0.5):
        res['illuminant_adjust'] = np.random.uniform(1 - 0.03, 1 + 0.03)
    else:
        res['illuminant_adjust'] = False

    if opt.isTrain and opt.face_color_transfer and (np.random.random() > 0.1):
        id = random.randint(0, len(opt.face_color) - 1)
        res['face_color'] = id
    else:
        res['face_color'] = False
    
    return res

def get_transform(opt, params, mode, method=transforms.InterpolationMode.BICUBIC, normalize=True):
    transform_list = [transforms.Lambda(lambda img: cv2.resize(img, params['osize'], interpolation=cv2.INTER_CUBIC))]

    if params['vflip']:
        if mode == 'input':
            transform_list.append(transforms.Lambda(lambda img: __flip(img)))
        elif mode == 'label':
            transform_list.append(transforms.Lambda(lambda img: __labelflip(img)))
    
    if params['face_color'] and mode == 'input':
        transform_list.append(transforms.Lambda(lambda img: __face_color_transfer(img, opt, params)))

    if params['resized_crop']:
        transform_list.append(transforms.Lambda(lambda img: __resized_crop(img, opt, params)))

    if params['padding_crop']:
        transform_list.append(transforms.Lambda(lambda img: __padding_crop(img, opt, params)))

    transform_list += [transforms.ToTensor()]

    if normalize and mode == 'input':
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    if params['illuminant_adjust'] and mode == 'input':
        transform_list.append(transforms.Lambda(lambda img: params['illuminant_adjust'] * img))
    
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __flip(img):
    return cv2.flip(img, 1)

def __labelflip(img):
    img = cv2.flip(img, 1)
    img[:,:,1] = 1 - img[:,:,1]
    return img

def __resized_crop(img, opt, params):
    x1, y1, x2, y2 = params['resized_crop']
    return cv2.resize(img[x1:x2,y1:y2,:], params['osize'], interpolation=cv2.INTER_CUBIC)

def __padding_crop(img, opt, params):
    x1, y1, x2, y2, top, bottom, left, right = params['padding_crop']
    return cv2.copyMakeBorder(img[x1:x2,y1:y2,:], top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

def __face_color_transfer(img, opt, params):
    return (face_color_transfer(opt.face_color[params['face_color']], img * 255) / 255.).astype(np.float32)