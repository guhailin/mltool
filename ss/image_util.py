import tensorflow as tf
from scipy import io
import scipy
from PIL import Image, ImageOps
import numpy as np


def get_image(path, resize='pad', out_dims=(224, 224)):
    img = Image.open(path).convert('RGB')

    if resize == 'resize':
        img = img.resize(out_dims)
    elif resize == 'pad':
        img = ImageOps.pad(img, out_dims)
    elif resize == None:
        pass
    else:
        raise ValueError("option for 'resize' not recognized")
    img = np.array(img)
    img = img/255.0
    return img


def get_label_png(path, resize='pad', out_dims=(224, 224)):
    img = Image.open(path)

    if resize == 'resize':
        img = img.resize(out_dims)
    elif resize == 'pad':
        img = ImageOps.pad(img, out_dims)
    elif resize == None:
        pass
    else:
        raise ValueError("option for 'resize' not recognized")

    arr = np.array(img)
    arr[arr == 255] = 0  # convert border class (255) to background class (0)

    return arr


def get_label_mat(path, resize='pad', out_dims=(224, 224)):
    mat = scipy.io.loadmat(path)
    arr = mat['GTcls']['Segmentation'][0, 0]

    img = Image.fromarray(arr, mode='P')

    if resize == 'resize':
        img = img.resize(out_dims)
    elif resize == 'pad':
        img = ImageOps.pad(img, out_dims)
    elif resize == None:
        pass
    else:
        raise ValueError("option for 'resize' not recognized")

    arr = np.array(img)

    return arr
