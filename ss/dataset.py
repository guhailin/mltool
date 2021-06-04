import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

import voc2012 as voc

import os

def voc2012():
    return voc.download_voc2012()

if __name__ == "__main__":
    print(voc2012())