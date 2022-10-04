from urllib.request import urlretrieve
from os.path import isfile, isdir
import utils
import tarfile
import os
import pickle
import numpy as np

label_names =  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

folder_path = 'data/cifar-10-batches-py'
zip_file = 'data/cifar-10-python.tar.gz'
download_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def load_cifar10_data():
    if not isdir(folder_path):
        if not isfile(zip_file):
            with utils.DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
                urlretrieve(download_url,zip_file,pbar.hook)
            print("Finish downloading!!!")
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
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path="data/")
            print("Finish extracting!!!")

def load_cifar10_batch(batch_id):
    assert batch_id in range(1,6)
    load_cifar10_data()
    batch_path = os.path.join(folder_path,'data_batch_'+str(batch_id))
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        # <class 'dict'>
        # dict_keys(['batch_label', 'labels', 'data', 'filenames'])
    images = batch['data']   # numpy.ndarray, shape = (10000, 3072)
    labels = batch['labels'] # python list
    images = images.reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)    #shape = (10000, 32, 32 , 3)
    labels = np.array(labels)
    return images, labels

def test():
    images, labels = load_cifar10_batch(1)
    print(images.shape)
    print(labels.shape)
    print([label_names[labels[i*10]] for i in range(9)])
    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.axis('off')
        plt.imshow(images[i*10])
    plt.show()
    batch = load_preprocess_batch_data(1)
    train_images = batch['train_images']
    train_labels = batch['train_labels']
    # train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_preprocess_batch(1)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.axis('off')
        plt.imshow(train_images[i*10])
    plt.show()
    print(train_images[0,0,0,:])
    print(np.max(train_images[0]))
    print(np.min(train_images[0]))

def preprocess_and_save(batch_id):
    images, labels = load_cifar10_batch(batch_id)
    images = utils.normalize_data(images)
    labels = utils.one_hot_encode(labels, 10)
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels =\
        utils.split_data(images, labels, train_size=0.8, valid_size=0.1, test_size=0.1)
    batch={'train_images':train_images, 'train_labels':train_labels,
            'valid_images':valid_images, 'valid_labels':valid_labels,
            'test_images':test_images, 'test_labels':test_labels}
    batch_path = os.path.join(folder_path,'preprocess_batch_'+str(batch_id))
    np.save(batch_path, np.asarray(batch))

def load_preprocess_batch_data(batch_id):
    assert batch_id in range(1,6)
    batch_path = os.path.join(folder_path,'preprocess_batch_'+str(batch_id)+'.npy')
    if not isfile(batch_path):
        preprocess_and_save(batch_id)
    batch = np.load(batch_path).item()
    # train_images = batch['train_images']
    # train_labels = batch['train_labels']
    # valid_images = batch['valid_images']
    # valid_labels = batch['valid_labels']
    # test_images = batch['test_images']
    # test_labels = batch['test_labels']
    # print('Train images shape : {}'.format(train_images.shape))
    # print('Train labels shape : {}'.format(train_labels.shape))
    # print('Valid images shape : {}'.format(valid_images.shape))
    # print('Valid labels shape : {}'.format(valid_labels.shape))
    # print('Test images shape : {}'.format(test_images.shape))
    # print('Test labels shape : {}'.format(test_labels.shape))
    # return train_images, train_labels, valid_images, valid_labels, test_images, test_labels
    return batch

def load_preprocess_batch(batch_id, batch_size):
    assert batch_id in range(1,6)
    batch_path = os.path.join(folder_path,'preprocess_batch_'+str(batch_id)+'.npy')
    if not isfile(batch_path):
        preprocess_and_save(batch_id)
    batch = np.load(batch_path).item()
    train_images = batch['train_images']
    train_labels = batch['train_labels']
    for start in range(0, train_labels.shape[0], batch_size):
        end = min(start + batch_size, train_labels.shape[0])
        yield train_images[start:end], train_labels[start:end]

