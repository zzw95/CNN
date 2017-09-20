from utils import DLProgress
from urllib.request import urlretrieve
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf

param_file = 'data/vgg16.npy'
VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self):
        # check param file if downloaded
        if not isfile(param_file):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='VGG16 Parameters') as pbar:
                urlretrieve('https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy'
                            ,param_file,pbar.hook)

        #load param from npy file, return python dictionary
        self.data_dict = np.load(param_file, encoding="latin1").item()
        print('vgg16 parameters loaded!')

    def build_nn(self, images):
        assert images.get_shape().as_list()[1:]==[224,224,3],"The shape of image should be [224,224,3]."

        #convert RGB to BGR
        red, green, blue = tf.split(images, num_or_size_splits=3, axis=3)
        BGR = tf.concat([blue-VGG_MEAN[0], green-VGG_MEAN[1], red-VGG_MEAN[2]], axis=3)
        assert BGR.get_shape().as_list()[1:]==[224,224,3]

        self.conv1_1 = self.conv_layer(BGR, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.max_pool_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.max_pool_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.max_pool_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.max_pool_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.max_pool_layer(self.conv5_3, 'pool5')

        pool5_shape = self.pool5.get_shape().as_list()
        self.flatten = tf.reshape(self.pool5, [-1,pool5_shape[1]*pool5_shape[2]*pool5_shape[3]])

        self.fc6 = self.fc_layer(self.flatten, 'fc6')
        assert self.fc6.get_shape().as_list()[1:]==[4096,]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, 'fc7')
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, 'fc8')
        self.prob = tf.nn.softmax(self.fc8, name='prob')


    def conv_layer(self, input, name):
        filter = self.get_weight(name)
        bias = self.get_bias(name)
        conv = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME', name=name)
        layer = tf.nn.relu(tf.nn.bias_add(conv, bias))
        print('{} : {}'.format(name, layer.get_shape()))
        return layer

    def max_pool_layer(self, input, name):
        layer = tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
        print('max_{} : {}'.format(name, layer.get_shape()))
        return layer

    def avg_pool_layer(self, input, name):
        layer =  tf.nn.avg_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
        print('avg_{} : {}'.format(name, layer.get_shape()))
        return layer

    def fc_layer(self, input, name):
        weight = self.get_weight(name)
        bias = self.get_bias(name)
        layer =  tf.add(tf.matmul(input, weight), bias)
        print('{} : {}'.format(name, layer.get_shape()))
        return layer

    def get_weight(self, name):
        return tf.constant(self.data_dict[name][0])

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1])


