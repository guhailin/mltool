import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
from .image_util import *
from .record_util import _bytes_feature, _float_feature, _int64_feature
import requests

def download_voc2012(IMG_WIDTH=224):
    file = downloadFILE('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz')
    os.system(f"tar -zxf {file}")

    path = 'benchmark_RELEASE/dataset/'

    train_record_path, val_record_path = write_record(path, IMG_WIDTH)
    return read_record(train_record_path, val_record_path)


def downloadFILE(url):
    print(f'downloading {url}')
    name = url.split('/')[-1]

    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print(name + "download finished!")
    return name


def write_record(path, IMG_WIDTH):
    train_txt=tf.io.gfile.GFile(path + 'train.txt').readlines()
    train_txt=[s.replace('\n', '') for s in train_txt]
    val_txt=tf.io.gfile.GFile(path + 'val.txt').readlines()
    val_txt=[s.replace('\n', '') for s in val_txt]

    train_record_file=f'voc2012_{IMG_WIDTH}_train.tfrecord'
    val_record_file=f'voc2012_{IMG_WIDTH}_val.tfrecord'
    _write_record(train_txt, path, train_record_file, IMG_WIDTH=IMG_WIDTH)
    _write_record(val_txt, path, val_record_file, IMG_WIDTH=IMG_WIDTH)
    return train_record_file, val_record_file


def read_record(train_path, val_path):

    train_dataset=tf.data.TFRecordDataset(train_path).map(_parse_function)

    val_dataset=tf.data.TFRecordDataset(val_path).map(_parse_function)
    return train_dataset, val_dataset

# 读取
_feature_description={
    'width':  tf.io.FixedLenFeature([], dtype=tf.int64, default_value=None),
    'height':  tf.io.FixedLenFeature([], dtype=tf.int64, default_value=None),
    'channel':  tf.io.FixedLenFeature([], dtype=tf.int64, default_value=None),
    'img':  tf.io.FixedLenFeature([], dtype=tf.string, default_value=None),
    'mat':  tf.io.FixedLenFeature([], dtype=tf.string, default_value=None),
}

def _parse_function(example_proto):
    parsed_record=tf.io.parse_single_example(
        example_proto, _feature_description)
    width=parsed_record['width']
    height=parsed_record['height']
    channel=parsed_record['channel']

    img=parsed_record['img']
    img=tf.io.decode_raw(img, out_type=tf.float64)
    img=tf.reshape(img, (width, height, channel))

    mat=parsed_record['mat']
    mat=tf.io.decode_raw(mat, out_type=tf.uint8)
    mat=tf.reshape(mat, (width, height))
    return (img, mat)


def _write_record(ids, path, record_file, IMG_WIDTH):
    with tf.io.TFRecordWriter(record_file) as writer:
        for id in tqdm(ids):
            img=path + 'img/' + id + '.jpg'
            mat=path + 'cls/' + id + '.mat'
            img=get_image(img, out_dims=(IMG_WIDTH, IMG_WIDTH))
            mat=get_label_mat(mat, out_dims=(IMG_WIDTH, IMG_WIDTH))

            features={
                "width": _int64_feature(IMG_WIDTH),
                "height": _int64_feature(IMG_WIDTH),
                "channel": _int64_feature(3),
                "img": _bytes_feature(img.tobytes()),
                "mat": _bytes_feature(mat.tobytes()),
            }
            example=tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
