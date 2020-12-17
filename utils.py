import os
import scipy
import scipy.ndimage
import glob
import imageio
import numpy as np
import h5py

from PIL import Image
from tqdm import tqdm
from config import Config

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

def make_data(data_array, label_array, cfg):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    # H * W * C to C * H * W
    data_array = np.transpose(data_array, (0, 3, 1, 2))
    label_array = np.transpose(label_array, (0, 3, 1, 2))
    save_folder = os.path.join(os.getcwd(), "checkpoint")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if cfg.is_train:
        savepath = os.path.join(save_folder, "train.h5")
    else:
        savepath = os.path.join(save_folder, "test.h5")
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data_array)
        hf.create_dataset('label', data=label_array)

def prepare_data(data_path, cfg):
    """
    Args:
        dataset: choose train dataset or test dataset
        
        For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if cfg.is_train:
        data_dir = os.path.join(os.getcwd(), data_path)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), data_path)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    return data

def input_setup():
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    cfg = Config()
    if cfg.is_train:
        data_path = cfg.dataset.train_path
    else:
        data_path = cfg.dataset.test_path
    file_list = prepare_data(data_path, cfg)
    padding = abs(cfg.image_size - cfg.label_size) / 2
    sub_input_array, sub_label_array = [], []
    if cfg.is_train:
        for i in tqdm(range(len(file_list))):
            input_, label_ = preprocess(file_list[i], cfg.scale)
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            for x in range(0, h - cfg.image_size+1, cfg.stride):
                for y in range(0, w-cfg.image_size+1, cfg.stride):
                    sub_input = input_[x:(x+cfg.image_size), y:(y+cfg.image_size)] # [33 x 33]
                    sub_label = label_[x+int(padding):x+int(padding)+cfg.label_size, y+int(padding):y+int(padding)+cfg.label_size] # [21 x 21]
                    # Make channel value
                    sub_input = sub_input.reshape([cfg.image_size, cfg.image_size, 1])  
                    sub_label = sub_label.reshape([cfg.label_size, cfg.label_size, 1])
                    sub_input_array.append(sub_input)
                    sub_label_array.append(sub_label)
    else:
        input_, label_ = preprocess(file_list[0], cfg.scale)
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0 
        for x in range(0, h-cfg.image_size+1, cfg.stride):
            nx += 1
            ny = 0
            for y in range(0, w-cfg.image_size+1, cfg.stride):
                ny += 1
                sub_input = input_[x:x+cfg.image_size, y:y+cfg.image_size] # [33 x 33]
                sub_label = label_[x+int(padding):x+int(padding)+cfg.label_size, y+int(padding):y+int(padding)+cfg.label_size] # [21 x 21]
                sub_input = sub_input.reshape([cfg.image_size, cfg.image_size, 1])  
                sub_label = sub_label.reshape([cfg.label_size, cfg.label_size, 1])
                sub_input_array.append(sub_input)
                sub_label_array.append(sub_label)
    data_array = np.array(sub_input_array)
    label_array = np.array(sub_label_array)
    make_data(data_array, label_array, cfg)
    if not cfg.is_train:
        return nx, ny

def merge(images, size):
    h, w = images.shape[2], images.shape[3]
    img = np.zeros((1, h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[:,j*h:(j+1)*h, i*w:(i+1)*w] = image
    return img

if __name__ == "__main__":
    input_setup()