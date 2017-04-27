import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


#import MNIST dataset from TF
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/dataset", one_hot=True) #loads the dataset each time it runs


#Load train data
x_train = mnist.train.images
y_train = mnist.train.labels

#Load test data
x_test = mnist.test.images
y_test = mnist.test.labels


x_train = x_train[:50000]
y_train = y_train[:50000]

#pring out the training data
print(x_train.shape)
print(y_train.shape)

#print out the testing data
print(x_test.shape)
print(y_test.shape)


def show_digit(index):
    label = y_train[index].argmax(axis=0) #the argmax gets the one_hot array and return the index where the 1 exists

    #get the 1D 784 array and reshape it into a 2D 28x28 array
    Reshaped_2D_array = x_train[index].reshape([28,28])
    # print(Reshaped_2D_array) This will print out the 28x28 array

    fig, axes = plt.subplots(1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.title("Training data, index: {}, Label: {}".format(index,label))
    plt.imshow(Reshaped_2D_array, cmap='Greens')
    plt.show()

def show_predicted_digit(image, pred, label):
    image = image.reshape([28, 28])
    plt.title("Original Image, Pred: {}, True label:{}".format(pred, label))
    plt.imshow(image)
    plt.show()

# show_digit(1)
# show_digit(2)
# show_digit(3)
#
#
# show_predicted_digit(x_train[1], 3,3)



batch_x, batch_y = mnist.train.next_batch(64)
print(batch_x.shape)

learning_rate = 0.001
training_epochs = 4
batch_size = 100
display_step = 1
model_path = "./talk_save/model1.ckpt"
alt_model_path = ".talk_save/model.ckpt"


n_input = 784 #MNIST data input(img shape: 28x28, flattened to be 784
n_hidden_1 = 384 #first layer number of nuerons
n_hidden_2 = 100 #2nd layer number of neurons
n_classes = 10 #MNIST classes for prediction(digits 0-9)


#the graph

tf.reset_default_graph()



with tf.name_scope("Inputs") as scope:
    x = tf.placeholder("float", [None, n_input], name='x_input')
    y = tf.placeholder("float", [None, n_classes], name='labels')


def multilayer_perceptron(x): #pass in the training set into x
    with tf.name_scope('hidden_01') as scope:
        #hidden layer 01 with RELU activation

        #weights and bias tensor
        h1weight = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1), name='h1_weights')
        h1bias = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1), name='b1_bias')


        #hidden layer 01 Operations
        layer_1 = tf.add(tf.matmul(x, h1weight), h1bias, name='Layer1_matmul') # (x* h1weight) + h1bias:: y = Wx +b
        layer_1 = tf.nn.relu(layer_1, name='Layer1_Relu') #activation Relu passes anything above 0 and blocks negatives


        #tensorboard histograms for layer 01
        tf.summary.histogram('weights_h1', h1weight)
        tf.summary.histogram('bias_h1', h1bias)


    with tf.name_scope('hidden_02') as scope:

        #hidden layer 02 with RELU activation
        h2weights = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1), name='h2_weights')
        h2bias = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1), name='b2_bias')

        layer_2 = tf.add(tf.matmul(layer_1, h2weights), h2bias, name='Layer2_add') #multiplies the layer and h2weights and adds h2bias:: y = Wx + b
        layer_2 = tf.nn.relu(layer_2, name='Layer2_Relu')


        #tensorboard histograms for layer 02
        tf.summary.histogram('weights_h2', h2weights)
        tf.summary.histogram('bias_h2', h2bias)


    with tf.name_scope('output_layer') as scope:
        #Logits layer with linear activation
        output_weights = tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.1), name='output_weights')
        output_bias = tf.Variable(tf.truncated_normal([n_classes], stddev=0.1), name='out_bias')
        logits_layer = tf.add(tf.matmul(layer_2, output_weights), output_bias, name='logits')    #here we create the equation y =Wx +b

    return logits_layer

pred = multilayer_perceptron(x) #we pass it the x placeholder created before the method above


#Create the loss
with tf.name_scope('cross_entropy'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

#create the optimizer
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) #optimizer makes changes to the weights and the biases
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) #uses the gradient descent

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#initialize the variables
init = tf.global_variables_initializer()


#create a saver to save and restore all the variables
saver = tf.train.Saver()


#run the graph on tensorboard

#file_writer = tf.summary.FileWriter('log_simple_graph/8', sess.graph)
"""
todo must revisit tensor board graph

"""




#Launch the graph and train the network by running the session

with tf.Session() as sess:
    sess.run(init)

    #training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        #Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            #run optimization op (backprop) and cost op (to get loss value)
            _, c,summary = sess.run([optimizer, loss, summary_op], )


































