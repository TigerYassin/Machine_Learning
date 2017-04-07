
"""
learn the details about tensorflow code
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/dataset', one_hot=True)


#get the data
x_data = mnist.train.images
y_data = mnist.train.labels

#get the testing data
x_test = mnist.test.images
y_test = mnist.test.labels

#cut down the training data
x_train = x_data[:50000]
y_train = y_data[:50000]


plt.imshow(x_test[int(input('enter an index'))].reshape([28,28]))
plt.show()








