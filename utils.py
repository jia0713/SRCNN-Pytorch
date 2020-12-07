import os
import scipy
import scipy.ndimage
import glob
import imageio
import numpy as np

from PIL import Image

def preprocess(path, scale=3):
    """
    (1) Read original images as Ycbcr format
    (2) Normalize
    (3) Apply image file with bicubic interpolation
    """
    image = imread(path)
    label_ = modcrop(image, scale)
    image = image / 255.0
    label_ = label_ / 255.0
    """
    Make the input image to low resolution and then back to its shale
    """
    input_ = scipy.ndimage.zoom(label_, (1./scale), prefilter=False)
    input_ = scipy.ndimage.zoom(input_, (scale/1.), prefilter=False)
    return input_, label_

def imread(path, is_grayscale=True):
    if is_grayscale:
        return imageio.imread(path, pilmode="YCbCr", as_gray=True).astype(np.float)
    else:
        return imageio.imread(path, pilmode="YCbCr").astype(np.float)

def modcrop(image, scale):
    assert(len(image.shape) == 3 or len(image.shape) == 2)
    h, w = image.shape[0], image.shape[1]
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    if len(image.shape) == 3:
        image = image[:h, :w, :]
    if len(image.shape) == 2:
        image = image[:h, :w]
    return image

# def input_setup():
#   """
#   Read image files and make their sub-images and saved them as a h5 file format.
#   """
#   sub_input_array, sub_label_array = [], []




if __name__ == "__main__":
    image = preprocess("Train/t1.bmp")
    print(image)
    print(image.shape)
