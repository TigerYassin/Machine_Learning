""""
here we use the mnist dataset in sklearn
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/datas")



#load the training data
x_data = mnist.train.images
y_data = mnist.train.labels




#load the testing data
x_test = mnist.test.images
y_test = mnist.test.labels



x_train = x_data[:50000]
y_train = y_data[:50000]

myTree = KNeighborsClassifier()
myTree.fit(x_train,y_train)
thisone = myTree.predict(x_test[[1]])
print(thisone)
plt.title('predicted 1')
mine = x_test[1].reshape([28,28])
plt.imshow(mine)
plt.show()

print(metrics.accuracy_score(myTree.predict(x_test), y_test))




print(x_data.shape)



