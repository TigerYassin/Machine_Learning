# import tensorflow as tf
#
# one = tf.constant(12)
# two = tf.constant()


"""
re-exploring scikitlearn
"""


from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics                 #for accuracy

iris = load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

myTree = tree.DecisionTreeClassifier()
myTree.fit(x_train, y_train)

prediction = myTree.predict(x_test)

print(metrics.accuracy_score(y_test, prediction))