import cv2
from glob import glob
import numpy as np


def preprocess_frame(directory, img_format="jpg", size=244,
                     drop_green=False, gray=False):

    img_format = "*." + img_format
    nb_images = len(glob(directory + img_format))

    num_channels = 3
    images = np.empty((nb_images, size, size, num_channels))
    for i, infile in enumerate(glob(directory + img_format)):
        img = cv2.imread(infile)
        if drop_green:
            img[:, :, 1] = 0
        if gray:
            num_channels = 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = square_pad(img)
        img = cv2.resize(img, (size, size))
        img = np.reshape(img, (1, size, size, num_channels))
        images[i, :, :, :] = img

    return images


def square_pad(img, padding_color=[0, 0, 0]):
    height = img.shape[0]
    width = img.shape[1]
    # find difference between longest side
    diff = np.abs(width - height)
    # amount of padding = half the diff between width and height
    pad_diff = diff // 2

    if height > width:
        # letter is longer than it is wide
        pad_top = 0
        pad_bottom = 0
        pad_left = pad_diff
        pad_right = pad_diff
        padded_img = cv2.copyMakeBorder(img,
                                        top=pad_top,
                                        bottom=pad_bottom,
                                        left=pad_left,
                                        right=pad_right,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=padding_color)
    elif width > height:
        # image is wide
        pad_top = pad_diff
        pad_bottom = pad_diff
        pad_left = 0
        pad_right = 0
        padded_img = cv2.copyMakeBorder(img,
                                        top=pad_top,
                                        bottom=pad_bottom,
                                        left=pad_left,
                                        right=pad_right,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=padding_color)
    elif width == height:
        padded_img = img.copy()

    return padded_img


def preprocess_for_vgg(img, size=224, color=True):
    img = cv2.resize(img, (size, size))
    x = np.array(img, dtype=float)
    x_fake_batch = x.reshape(1, *x.shape)
    x = x_fake_batch
    if color:
        # Zero-center by mean pixel
        x[:, :, :, 2] -= 123.68
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 0] -= 103.939
    return x


def edit_bg(img, bg_img_path):
    img_front = img.copy()
    img_back = cv2.imread(bg_img_path)
    height, width = img_front.shape[:2]
    resize_back = cv2.resize(img_back, (width, height), interpolation=cv2.INTER_CUBIC)
    for i in range(width):
        for j in range(height):
            pixel = img_front[j, i]
            if np.all(pixel == [0, 0, 0]):
                img_front[j, i] = resize_back[j, i]
    return img_front

