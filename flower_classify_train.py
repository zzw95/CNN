import utils
import flower_photos_data
import vgg16
import tensorflow as tf
import numpy as np
from os.path import isfile, isdir
import os

label_dict ={0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
def process_data():
    images, labels, label_dict = flower_photos_data.load_flower_datasets()
    assert np.max(images[0])<=1, 'The image should be scaled to 0-1'
    images, labels = utils.shuffle_data(images, labels)
    labels_onehot = utils.one_hot_encode(labels)
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = \
        utils.split_data(images, labels_onehot, train_size=0.8, valid_size=0.1, test_size=0.1)
    os.mkdir('data/flower_npy')
    np.save('data/flower_npy/train_images.npy', train_images)
    np.save('data/flower_npy/train_labels.npy', train_labels)
    np.save('data/flower_npy/valid_images.npy', valid_images)
    np.save('data/flower_npy/valid_labels.npy', valid_labels)
    np.save('data/flower_npy/test_images.npy', test_images)
    np.save('data/flower_npy/test_labels.npy', test_labels)

if not isdir('data/flower_npy'):
    process_data()
train_images = np.load('data/flower_npy/train_images.npy')
train_labels = np.load('data/flower_npy/train_labels.npy')
valid_images = np.load('data/flower_npy/valid_images.npy')
valid_labels = np.load('data/flower_npy/valid_labels.npy')
print("Train images: {}, labels: {}".format(train_images.shape, train_labels.shape))
print("Valid images: {}, labels: {}".format(valid_images.shape, valid_labels.shape))

inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,3], name='inputs')
labels_ = tf.placeholder(dtype=tf.float32, shape=[None,len(label_dict)], name='labels')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

vgg = vgg16.Vgg16()
vgg.build_nn(inputs_)

hidden1 = tf.contrib.layers.fully_connected(vgg.flatten, 1024, activation_fn = tf.nn.relu)
drop1 = tf.nn.dropout(hidden1, keep_prob=keep_prob)
hidden2 = tf.contrib.layers.fully_connected(drop1, 256, activation_fn = tf.nn.relu)
drop2 = tf.nn.dropout(hidden2, keep_prob=keep_prob)
logits = tf.contrib.layers.fully_connected(drop2, len(label_dict), activation_fn = None)
prediction = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))

compare = tf.equal(tf.argmax(prediction,axis=1), tf.argmax(labels_, axis=1))
accuracy = tf.reduce_mean(tf.cast(compare, dtype=tf.float32))

optimizer = tf.train.AdadeltaOptimizer().minimize(cost)

saver = tf.train.Saver()

epoches = 10
iteration = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Start training ...... ')
    for e in range(epoches):
        for batch_images, batch_labels in utils.get_batches(train_images, train_labels, n_batches=10):
            iteration += 1
            loss,  _ = sess.run([cost,  optimizer],
                               feed_dict={inputs_:batch_images, labels_:batch_labels, keep_prob:0.5})
            print("Epoch: {}/{}".format(e+1, epoches),
                  "Iteration: {}".format(iteration),
                  "Train loss: {:.5f}".format(loss))
            if iteration%10==0:
                val_acc = sess.run(accuracy, feed_dict={inputs_:valid_images, labels_:valid_labels, keep_prob:1.0})
                print("-------------------")
                print("Epoch: {}/{}".format(e+1, epoches),
                      "Iteration: {}".format(iteration),
                      "Validation Acc: {:.4f}".format(val_acc))
                print("-------------------")

    saver.save(sess, "data/flower_checkpoints/flowers.ckpt")


