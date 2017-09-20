import utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16

img_tiger = utils.load_image('data/tiger.jpeg')
img_puzzle = utils.load_image('data/puzzle.jpeg')
img_tiger = utils.resize_image(img_tiger).reshape([1, 224, 224, 3])*255
img_puzzle = utils.resize_image(img_puzzle).reshape([1,224, 224, 3])*255

images = np.concatenate([img_tiger, img_puzzle], axis =0)

with tf.Session() as sess:
    images_ = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.build_nn(images_)
    results = sess.run(vgg.prob, feed_dict={images_:images})
    print(results)
    print('tiger-------------')
    utils.print_prob(results[0], 'data/synset.txt')
    print('puzzle------------')
    utils.print_prob(results[1], 'data/synset.txt')

