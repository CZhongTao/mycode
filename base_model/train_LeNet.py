#//usr/bin/env python
#coding=utf-8

"""
file: train_LeNet.py
autor: chenzhongtao
"""

import numpy as np
import tensorflow as tf
import argparse
import input_data


def setup_model(input_x, input_y):
    """
    setup cnn model
    """
    # conv1 input_x shape: (batch_size, 28, 28, 1), filter number: 8 filter size: 3*3
    conv1 = tf.layers.conv2d(input_x, 8, 3, activation=tf.nn.relu)
    # 2*2 max pooling 
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(conv1, 16, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    flatten = tf.layers.flatten(conv2)
    dense1 = tf.layers.dense(flatten, 20, activation=tf.nn.relu)
    output = tf.layers.dense(dense1, 10, activation=tf.sigmoid)

    loss = loss_fuc(output, input_y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    min_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    accuracy = eval(output, input_y)
    return [loss, accuracy, min_op]


def loss_fuc(y_predict, y_true):
    """
    loss function
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_true))


def eval(y_predict, y_true):
    """
    get accuracy
    """
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def train_model(mnist):
    """
    train and test model
    """
    epoch = 100
    batch_size = 128
    
    input_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    input_y = tf.placeholder(tf.float32, [None, 10])

    [loss, accuracy, train_step] = setup_model(input_x, input_y)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    feature = mnist.train.images
    label = mnist.train.labels
    feature = feature.reshape((feature.shape[0], 28, 28, 1))
    steps = int(feature.shape[0] / batch_size)

    for i in range(epoch):
        loss_list = []
        accuracy_list = []
        for j in range(steps):
            batch = [feature[j: j + batch_size], label[j: j + batch_size]]
            loss_out, accuracy_out, _ = sess.run([loss, accuracy, train_step], feed_dict={input_x: batch[0], input_y: batch[1]})
            loss_list.append(loss_out)
            accuracy_list.append(accuracy_out)
        print ("epoch: {}, loss mean: {}, accuracy: {}".format(i, np.mean(np.array(loss_list)), np.mean(np.array(accuracy_list))))

    print ("test model")
    test_feature = mnist.test.images
    test_label = mnist.test.labels
    test_feature = test_feature.reshape((test_feature.shape[0], 28, 28, 1))
    test_accuracy = sess.run(accuracy, feed_dict={input_x: test_feature, input_y: test_label})
    print ("test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tensorflow model training and testing')
    parser.add_argument('data_dir', type=str, help='data directory contains training samples')
    args = parser.parse_args()

    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    train_model(mnist)
