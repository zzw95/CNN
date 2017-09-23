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

def shuffle_data(datasets, labels):
    # 随机化处理
    assert datasets.shape[0]==labels.shape[0]
    permutation = np.random.permutation(labels.shape[0])
    shuffled_datasets = datasets[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_datasets, shuffled_labels

def normalize_data(datasets):
    min_ = np.min(datasets)
    max_ = np.max(datasets)
    return (datasets-min_)/(max_-min_)

def split_data(datasets, labels, train_size, valid_size, test_size):
    assert (train_size+valid_size+test_size)<=1
    assert datasets.shape[0]==labels.shape[0]
    num = datasets.shape[0]
    train_idx = int(num*train_size)
    valid_idx = int(num*(train_size+valid_size))
    test_idx = int(num*(train_size+valid_size+test_size))
    train_datasets = datasets[:train_idx]
    train_labels = labels[:train_idx]
    valid_datasets = datasets[train_idx:valid_idx]
    valid_labels = labels[train_idx:valid_idx]
    test_datasets = datasets[valid_idx:test_idx]
    test_labels = labels[valid_idx:test_idx]
    return train_datasets, train_labels, valid_datasets, valid_labels, test_datasets, test_labels

def one_hot_encode(labels, dim = None):
    """
    :param: Y -- vector containing the labels, shape = (number of examples,1) or (number of examples)
    :return: Y_onehot -- one hot matrix, shape = (dim, number of examples)
    """
    if dim==None:
        dim = np.max(labels) + 1
    labels_onehot = ( np.arange(dim) == labels.reshape([labels.shape[0],1]) ) * 1

    # or
    # from tflearn.data_utils import to_categorical
    # return to_categorical(labels,dim)

    assert labels_onehot.shape==(labels.shape[0], dim)
    return labels_onehot

def get_batches(datasets, labels, n_batches):
    batch_size = len(datasets) // n_batches
    for i in range(n_batches):
        batch_datasets = datasets[i*batch_size:(i+1)*batch_size]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]
        yield  batch_datasets, batch_labels



