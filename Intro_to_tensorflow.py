import tensorflow as tf
print(tf.__version__)

tf.reset_default_graph()
graph = tf.get_default_graph()

sess = tf.Session()


input_a = tf.constant(4., dtype=tf.float32, name='a')
input_b = tf.constant(3., dtype=tf.float32, name='b')

c = tf.add(input_a, input_b, name='a_plus_b')
d = tf.multiply(input_a, input_b, name='a_mul_b')

e = tf.multiply(c, d, name='c_mul_d')

operations = graph.get_operations()
for operation in operations:
    print(operation.name, " ", operation.type)

print(operations[-2].node_def)
print(sess.run(c))

summary_writer = tf.summary.FileWriter('log_simple', sess.graph)
print(sess.run(e))

# !tensorboard --log=log_simple


