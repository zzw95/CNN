from urllib.request import urlretrieve
from os.path import isfile, isdir
import utils
import tarfile
import os
import numpy as np

photos_folder = 'data/flower_photos'
zip_file = 'data/flower_photos.tar.gz'
download_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
npy_file = 'data/flower_photos.npy'

def load_flower_datasets(limits=200):
    if not isdir(photos_folder):
        if not isfile(zip_file):
            with utils.DLProgress(unit='B', unit_scale=True, miniters=1, desc='Flowers Dataset') as pbar:
                urlretrieve(download_url,zip_file,pbar.hook)
        with tarfile.open(zip_file) as tar:
           def is_within_directory(directory, target):
               
               abs_directory = os.path.abspath(directory)
               abs_target = os.path.abspath(target)
           
               prefix = os.path.commonprefix([abs_directory, abs_target])
               
               return prefix == abs_directory
           
           def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
           
               for member in tar.getmembers():
                   member_path = os.path.join(path, member.name)
                   if not is_within_directory(path, member_path):
                       raise Exception("Attempted Path Traversal in Tar File")
           
               tar.extractall(path, members, numeric_owner) 
               
           
           safe_extract(tar, path="data/")

    label_dict={}
    images = []
    labels = []
    for index,type in enumerate(os.listdir(photos_folder)):
        path = os.path.join(photos_folder, type)
        if not isdir(path):
            continue
        num=0
        label_dict[index] = type
        print('Loading {} images .......'.format(type))
        for file in os.listdir(path):
            num+=1
            if num>limits:
                break
            file = os.path.join(photos_folder,type, file)
            image = utils.load_image(file)
            image = utils.resize_image(image)
            images.append([image,])
            labels.append(index)
    images = np.concatenate(images, axis=0)
    labels = np.array(labels)

    print('All {} images loaded!'.format(images.shape[0]))
    print('images shape : {}'.format(images.shape))
    print('labels shape : {}'.format(labels.shape))

    return images, labels, label_dict

def save_flower_datasets():
    images, labels, label_dict = load_flower_datasets()
    save = {'images':images, 'labels':labels, 'label_dict':label_dict}
    np.save(npy_file, np.asarray(save))     # too large
