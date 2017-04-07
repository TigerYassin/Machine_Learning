import tensorflow as tf

tf.reset_default_graph()
graph = tf.get_default_graph()
sess = tf.Session()


tensor_a = tf.constant([[4,5,6], [1,3,5], [3,1,3]], shape=[3,3], dtype=tf.float32, name='tensor_b')
tensor_b = tf.constant([[4,3,5], [12,3,45], [63,41, 3]], shape=[3,3], dtype=tf.float32, name='tensor_b')

tensor_matrix_mul = tf.matmul(tensor_a, tensor_b)

print(tensor_a.get_shape())
print(sess.run(tensor_matrix_mul))

summary_writer = tf.summary.FileWriter('hey/matrix_graph', sess.graph)
