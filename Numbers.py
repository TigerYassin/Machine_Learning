import tensorflow as tf
import numpy as np


#make the fake data
X_data = np.random.rand(100).astype(np.float32)
fake_Weight = 10
fake_Bias = 3

y_data = (X_data * fake_Weight) + fake_Bias


#make the structure with tensorflow
Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

y = (X_data * Weight) + bias

# get the loss
loss = tf.reduce_mean(tf.square(y-y_data))

#get the trainer and the optimizer
train = tf.train.GradientDescentOptimizer(.2)
optimize = train.minimize(loss)



#initialize the variables and start the session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for x in range(50):
        sess.run(optimize)
        if x % 10 ==0:
            print("Fake Weight: ", fake_Weight)
            print(sess.run(Weight))
            print("Fake Bias: ", fake_Bias)
            print(sess.run(bias))

            print()