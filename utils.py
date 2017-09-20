from tqdm import tqdm
from  scipy import ndimage
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np

class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def load_image(path):
    # return ndimage.imread(path)
    return skimage.io.imread(path)

def resize_image(image, shape=[224,224]):
    return skimage.transform.resize(image, shape)

def cut_square_image(image):
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    cut_img = image[yy: yy + short_edge, xx: xx + short_edge]
    return cut_img

def print_prob(prob, file_path, n=5):
    with open(file_path) as f:
        synset = [l.strip() for l in f.readlines()]
    pred = np.argsort(prob)[::-1]
    topN = [(synset[pred[i]], prob[pred[i]]) for i in range(n)]
    for p,s in topN:
        print('prob:{} -> {}'.format(p,s))
    return topN


