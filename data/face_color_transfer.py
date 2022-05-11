# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import os
import colour
import time
import cv2

def RGB2YCbCr(rgb):
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    Y = 0.257*R+0.504*G+0.098*B+16
    Cb = -0.148*R-0.291*G+0.439*B+128
    Cr = 0.439*R-0.368*G-0.071*B+128

    return np.dstack([Y, Cb, Cr])

def count(w):
    return dict(zip(*np.unique(w, return_counts=True)))


def count_array(w, size):
    d = count(w)
    return np.array([d.get(i, 0) for i in range(size)])


def get_border(Sa):
    si = np.argmax(Sa)
    t1 = si - 1
    t2 = si + 1
    diff = 0
    while t1 >= 0 and t2 <= 255:
        diff += (Sa[t1] - Sa[t2])
        if abs(diff) > 2 * max(Sa[t1], Sa[t2]) or Sa[t1] == 0 or Sa[t2] == 0:
            # print("Sa", Sa[t1], Sa[t2])
            return [t1, t2]
        t1 -= 1
        t2 += 1
    t1 = max(0, t1)
    t2 = min(255, t2)
    return [t1, t2]


def deal(rgb):
    y = RGB2YCbCr(rgb)
    b = (y[:, :, 1] >= 77) & (y[:, :, 1] <= 127) & (
        y[:, :, 2] >= 133) & (y[:, :, 2] <= 194)
    XYZ = colour.sRGB_to_XYZ(rgb / 255, apply_cctf_decoding=False)
    LAB = colour.XYZ_to_Lab(XYZ)
    lab = np.round(LAB).astype(np.int64)
    # a, b += 128
    lab[:, :, 1:3] += 128
    # 0 ~ 255
    Sa = count_array(lab[:, :, 1][b], 256)
    Sb = count_array(lab[:, :, 2][b], 256)
    SaBorder = get_border(Sa)
    SbBorder = get_border(Sb)
    b2 = (((lab[:, :, 1] >= SaBorder[0]) & (lab[:, :, 1] <= SaBorder[1])) | (
        (lab[:, :, 2] >= SbBorder[0]) & (lab[:, :, 2] <= SbBorder[1])))
    # plt.subplot(121)
    # plt.imshow(b, "gray")
    # plt.subplot(122)
    # plt.imshow(b2, "gray")
    # plt.show()
    return lab, b2, Sa, Sb, SaBorder, SbBorder, np.mean(lab[:, :, 1][b2]), np.mean(lab[:, :, 2][b2])

def get_info(rgb):
    y = RGB2YCbCr(rgb)
    b = (y[:, :, 1] >= 77) & (y[:, :, 1] <= 127) & (
        y[:, :, 2] >= 133) & (y[:, :, 2] <= 194)
    XYZ = colour.sRGB_to_XYZ(rgb / 255, apply_cctf_decoding=False)
    LAB = colour.XYZ_to_Lab(XYZ)
    lab = np.round(LAB).astype(np.int64)
    # a, b += 128
    lab[:, :, 1:3] += 128
    # 0 ~ 255
    Sa = count_array(lab[:, :, 1][b], 256)
    Sb = count_array(lab[:, :, 2][b], 256)
    SaBorder = get_border(Sa)
    SbBorder = get_border(Sb)
    info = {'Sa': Sa, 'Sb': Sb, 'SaBorder': SaBorder, 'SbBorder': SbBorder}
    return info

def informed_deal(rgb, rgb_info):
    XYZ = colour.sRGB_to_XYZ(rgb / 255, apply_cctf_decoding=False)
    LAB = colour.XYZ_to_Lab(XYZ)
    lab = np.round(LAB).astype(np.int64)
    # a, b += 128
    lab[:, :, 1:3] += 128
    # 0 ~ 255
    Sa = rgb_info['Sa']
    Sb = rgb_info['Sb']
    SaBorder = rgb_info['SaBorder']
    SbBorder = rgb_info['SbBorder']
    b2 = (((lab[:, :, 1] >= SaBorder[0]) & (lab[:, :, 1] <= SaBorder[1])) | (
        (lab[:, :, 2] >= SbBorder[0]) & (lab[:, :, 2] <= SbBorder[1])))
    # plt.imshow(b2, "gray")
    # plt.show()
    return lab, b2, Sa, Sb, SaBorder, SbBorder, np.mean(lab[:, :, 1][b2]), np.mean(lab[:, :, 2][b2])

def informed_face_color_transfer(source, source_info, target, target_info):
    slab, sb, Sa, Sb, [sab, sae], [sbb, sbe], sam, sbm = informed_deal(source, source_info)
    tlab, tb, Ta, Tb, [tab, tae], [tbb, tbe], tam, tbm = informed_deal(target, target_info)

    sam = (sab + sae) / 2.0
    sbm = (sbb + sbe) / 2.0
    tam = (tab + tae) / 2.0
    tbm = (tbb + tbe) / 2.0

    # plt.plot(Sa, 'r.')
    # plt.plot(Ta, 'r*')
    # plt.plot(Sb, 'b.')
    # plt.plot(Tb, 'b*')
    # plt.show()

    rsa1 = (sam - sab) * 1.0 / (tam - tab)
    rsa2 = (sae - sam) * 1.0 / (tae - tam)
    rsb1 = (sbm - sbb) * 1.0 / (tbm - tbb)
    rsb2 = (sbe - sbm) * 1.0 / (tbe - tbm)

    def transfer(a, sam, tam, rsa1, rsa2, sab, sae):
        # aold = a.copy()
        b = a < tam
        a[b] = rsa1 * (a[b] - tam) + sam
        a[~b] = rsa2 * (a[~b] - tam) + sam
        # Correction
        b1 = (a < sab) & (a > sab - 2)
        b2 = (a > sae) & (a < 2 + sae)
        # b3 = (a > sab) & (a < sae)
        # b4 = ~(b1 | b2 | b3)
        a[b1] = sab
        a[b2] = sae
        # print(np.sum(b1), np.sum(b2), np.sum(b3), np.sum(b4))
        #a[b4] = aold[b4]
        return a

    # plt.subplot(121)
    # plt.imshow(sb, "gray")
    # plt.subplot(122)
    # plt.imshow(tb, "gray")
    # plt.show()

    tlab[:, :, 1][tb] = transfer(
        tlab[:, :, 1][tb], sam, tam, rsa1, rsa2, sab, sae)
    tlab[:, :, 2][tb] = transfer(
        tlab[:, :, 2][tb], sbm, tbm, rsb1, rsb2, sbb, sbe)
    tlab[:, :, 1:3] -= 128
    tlab[:, :, 1:3] = np.clip(tlab[:, :, 1:3], -128, 128)
    # return Lab2RGB(tlab)
    XYZ = colour.Lab_to_XYZ(tlab)
    RGB = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False) * 255
    return RGB


def compressor(x):
    if x < 0.8:
        return x
    else:
        return 0.8 + 2.0 * np.arctan(x - 1) / np.pi

def face_color_transfer(source, target):
    slab, sb, Sa, Sb, [sab, sae], [sbb, sbe], sam, sbm = deal(source)
    tlab, tb, Ta, Tb, [tab, tae], [tbb, tbe], tam, tbm = deal(target)

    sam = (sab + sae) / 2.0
    sbm = (sbb + sbe) / 2.0
    tam = (tab + tae) / 2.0
    tbm = (tbb + tbe) / 2.0

    # plt.plot(Sa, 'r.')
    # plt.plot(Ta, 'r*')
    # plt.plot(Sb, 'b.')
    # plt.plot(Tb, 'b*')
    # plt.show()

    rsa1 = (sam - sab) * 1.0 / (tam - tab)
    rsa2 = (sae - sam) * 1.0 / (tae - tam)
    rsb1 = (sbm - sbb) * 1.0 / (tbm - tbb)
    rsb2 = (sbe - sbm) * 1.0 / (tbe - tbm)
    # print("Trans Params", rsa1, rsa2, rsb1, rsb2)
    rsa1 = compressor(rsa1)
    rsa2 = compressor(rsa2)
    rsb1 = compressor(rsb1)
    rsb2 = compressor(rsb2)

    # rmax = max(rsa1, max(rsa2, max(rsb1, rsb2)))
    # if rmax > 0.6:
    #     rmax = rmax * 0.6
    #     rsa1 = rsa1 / rmax
    #     rsa2 = rsa2 / rmax
    #     rsb1 = rsb1 / rmax
    #     rsb2 = rsb2 / rmax

    # rsa1 = min(0.5, rsa1)
    # rsa2 = min(0.5, rsa2)
    # rsb1 = min(0.5, rsb1)
    # rsb2 = min(0.5, rsb2)

    # if (sae - tam) < 0:
    #     rsa1 = max((255 - sam) / (sae - tam), rsa1)
    #     rsa2 = max((255 - sam) / (sae - tam), rsa2)
    # else:
    #     rsa1 = min((255 - sam) / (sae - tam), rsa1)
    #     rsa2 = min((255 - sam) / (sae - tam), rsa2)

    # if (sbe - tbm) < 0:
    #     rsb1 = max((255 - sbm) / (sbe - tbm), rsb1)
    #     rsb2 = max((255 - sbm) / (sbe - tbm), rsb2)
    # else:
    #     rsb1 = min((255 - sbm) / (sbe - tbm), rsb1)
    #     rsb2 = min((255 - sbm) / (sbe - tbm), rsb2)

    # print("Justified Trans Params", rsa1, rsa2, rsb1, rsb2)

    def transfer(a, sam, tam, rsa1, rsa2, sab, sae):
        aold = a.copy()
        b = a < tam
        a[b] = rsa1 * (a[b] - tam) + sam
        a[~b] = rsa2 * (a[~b] - tam) + sam
        # Correction
        b1 = (a < sab) & (a > sab - 2)
        b2 = (a > sae) & (a < 2 + sae)
        b3 = (a > sab) & (a < sae)
        b4 = ~(b1 | b2 | b3)
        a[b1] = sab
        a[b2] = sae
        #print(np.sum(b1), np.sum(b2), np.sum(b3), np.sum(b4))
        #a[b4] = aold[b4]
        return a

    # plt.subplot(121)
    # plt.imshow(sb, "gray")
    # plt.subplot(122)
    # plt.imshow(tb, "gray")
    # plt.show()

    tlab[:, :, 1][tb] = transfer(
        tlab[:, :, 1][tb], sam, tam, rsa1, rsa2, sab, sae)
    tlab[:, :, 2][tb] = transfer(
        tlab[:, :, 2][tb], sbm, tbm, rsb1, rsb2, sbb, sbe)
    tlab[:, :, 1:3] -= 128
    tlab[:, :, 1:3] = np.clip(tlab[:, :, 1:3], -128, 128)
    # return Lab2RGB(tlab)
    XYZ = colour.Lab_to_XYZ(tlab)
    RGB = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False) * 255
    return RGB