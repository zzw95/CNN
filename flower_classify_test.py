import vgg16
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

label_dict ={0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}

test_images = np.load('data/flower_npy/test_images.npy')
test_labels = np.load('data/flower_npy/test_labels.npy')
print("Test images: {}, labels: {}".format(test_images.shape, test_labels.shape))

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

# optimizer = tf.train.AdadeltaOptimizer().minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:
    print('Restoring parameters ......')
    saver.restore(sess, tf.train.latest_checkpoint('data/flower_checkpoints'))
    predict, acc = sess.run([prediction, accuracy], feed_dict={inputs_:test_images, labels_:test_labels, keep_prob:1.0})
    print(acc)

print('The fist 9 images: {}'.format([label_dict[i] for i in np.argmax(test_labels[:9],axis=1)]))
print('Prediction: {}'.format([label_dict[i] for i in np.argmax(predict[:9],axis=1)]))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(test_images[i])
plt.show()
