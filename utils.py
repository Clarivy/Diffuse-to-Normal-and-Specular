import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
from skimage import io
import imageio

def adjust_gamma(image, gamma=1.0, EPS = 0.01):
    if np.abs(gamma - 1) < EPS:
        return image
    invGamma = 1.0 / gamma
    if image.dtype == np.uint16:
        return (cv2.pow(image / 65535, invGamma) * 65535).astype(np.uint16)
    if image.dtype == np.float32:
        return (cv2.pow(image, invGamma))
    if image.dtype == np.float64:
        return (cv2.pow(image, invGamma))
    if image.dtype == np.uint8:
        return (cv2.pow(image / 255, invGamma) * 255).astype(np.uint8)
    raise Exception("adjust_gamma: image must be uint16")
	# apply gamma correction

def readImage(image, debug = False, gamma = 1.0, dtype = np.uint16):
    img = imageio.imread(image)

    if img is None:
        raise Exception("Image not found while reading" + image)

    img = adjust_gamma(img, gamma=gamma)
    if dtype == np.uint16:
        if img.dtype == np.uint8:
            img = img.astype(np.uint16) * 256
        elif img.dtype == np.float16 or img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 65535).astype(np.uint16)
        elif img.dtype != np.uint16:
            raise Exception("Read Image: unknown type")
        
        if debug:
            plt.matshow((img / 63335)[:,:,[2,1,0]])
            plt.show()

    elif dtype == np.float32:
        if img.dtype == np.uint8:
            img = (img / 255).astype(np.float32)
        elif img.dtype == np.uint16:
            img = (img / 65535).astype(np.float32)
        elif img.dtype == np.float16 or img.dtype == np.float32 or img.dtype == np.float64:
            img = img
        else:
            raise Exception("Read Image: unknown type")

        if debug:
            plt.matshow(img[:,:,[2,1,0]])
            plt.show()

    elif dtype == np.uint8:
        if img.dtype == np.uint16:
            img = (img.astype(np.float64) / 256).astype(np.uint8)
        elif img.dtype == np.float16 or img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            raise Exception("Read Image: unknown type")
        
        if debug:
            plt.matshow((img / 255)[:,:,[2,1,0]])
            plt.show()
    else:
        raise Exception("Read Image: unsupported type")
    
    return img

def calc_loss(img, img_ref, debug = False, display = False):
    img_ref = img_ref.astype(np.float32)
    img = img.astype(np.float32)
    delta_img = np.abs(img-img_ref)

    img_size = img.shape[0] * img.shape[1]
    loss = np.sum(np.sum(delta_img, axis=0), axis=0) / img_size
    std = np.std(delta_img)
    max_delta = np.max(delta_img)
    min_delta = np.min(delta_img)

    if debug:
        print("Loss: ", loss)
        print("Std: ", std)
        print("Max delta: ", max_delta)
        print("Min delta: ", min_delta)

    if display:
        temp_img = img - img_ref
        temp_img[0:100,0:100,:] = np.array([-1, -1, -1])
        temp_img[0:100,100:200,:] = np.array([0, 0, 0])
        temp_img[0:100,200:300,:] = np.array([1, 1, 1])
        plt.subplot(2, 2, 1)
        plt.matshow(delta_img.sum(axis=2), cmap=cm.Spectral_r, fignum=False)
        plt.subplot(2, 2, 2)
        plt.matshow(temp_img[:, :, 0], cmap=cm.Blues, fignum=False)
        plt.subplot(2, 2, 3)
        plt.matshow(temp_img[:, :, 1], cmap=cm.Greens, fignum=False)
        plt.subplot(2, 2, 4)
        plt.matshow(temp_img[:, :, 2], cmap=cm.Reds, fignum=False)

        plt.figure(2)
        temp_img = copy.deepcopy(img)
        temp_img[temp_img < 1] = 0
        temp_img[(temp_img >= 1)] = temp_img[(temp_img >= 1)] - 1
        temp_img *= 100
        plt.subplot(2, 2, 1)
        plt.matshow(temp_img, cmap=cm.Spectral_r, fignum=False)
        plt.subplot(2, 2, 2)
        plt.matshow(temp_img[:, :, 0], cmap=cm.Blues, fignum=False)
        plt.subplot(2, 2, 3)
        plt.matshow(temp_img[:, :, 1], cmap=cm.Greens, fignum=False)
        plt.subplot(2, 2, 4)
        plt.matshow(temp_img[:, :, 2], cmap=cm.Reds, fignum=False)

        plt.figure(3)
        temp_img = copy.deepcopy(img)
        temp_img[temp_img > 0] = 0
        temp_img = -temp_img
        temp_img *= 100
        plt.subplot(2, 2, 1)
        plt.matshow(temp_img, cmap=cm.Spectral_r, fignum=False)
        plt.subplot(2, 2, 2)
        plt.matshow(temp_img[:, :, 0], cmap=cm.Blues, fignum=False)
        plt.subplot(2, 2, 3)
        plt.matshow(temp_img[:, :, 1], cmap=cm.Greens, fignum=False)
        plt.subplot(2, 2, 4)
        plt.matshow(temp_img[:, :, 2], cmap=cm.Reds, fignum=False)
        plt.show()
    return loss

def read_chart_color(chart_color_path, rows = 4, columns = 6, channels = 3, dtype = np.uint8):
    data = []
    with open(chart_color_path) as f:
        data = f.read()
    data = list(map(float, re.split(' |\n|\t', data.strip())))
    data = np.array(data, dtype=dtype).reshape(rows, columns, channels)
    return data