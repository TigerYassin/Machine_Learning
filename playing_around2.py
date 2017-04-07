"""
here we play with tensorflow
"""

import tensorflow as tf
import numpy as np

a = tf.constant([   [[4,0,2], [1,2,6]],   [ [4,0,2], [1,2,6]]])
b = tf.Variable([   [[2, 3, 1], [3,4,5]], [[2, 3, 1], [3,4,5]]])

myNP = np.random.rand(10)
# mytf = tf.random_normal([[5],[2]])
mytf = tf.random_uniform([1])


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# print(sess.run(a * b))
#
#
# print(sess.run(a))

# print(myNP)
print(sess.run(mytf))