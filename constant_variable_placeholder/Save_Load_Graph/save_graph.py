import tensorflow as tf
import numpy as np

#Let define a function Y = f(X) = X[1] + 2*X[2] + 3*X[3] + 0.5
n_features = 3
W = tf.Variable(np.array([1,2,3]).reshape(1,-1), dtype = tf.float32, name = 'coef')
X = tf.placeholder(tf.float32, [None, n_features], name = 'X')
b = tf.Variable(0.5,name = 'bias')
Y = tf.add(tf.matmul(X, tf.transpose(W)), b, name = 'Y')

with tf.Session() as sess:
    # Remember to save the graph right after the function's definition. Otherwise, you may get some error when restoring them later.
    tf.train.write_graph(sess.graph_def, '.', 'graph_def', False)   
    # Initialize variables before saving their values.
    sess.run(tf.initialize_all_variables())
    # Save all the variables.
    saver = tf.train.Saver(tf.all_variables())
    saver.save(sess, 'checkpoint-data')

