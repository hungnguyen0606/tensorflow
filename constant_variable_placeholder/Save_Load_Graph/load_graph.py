import tensorflow as tf
import numpy as np

with tf.Session() as sess:
	with open('graph_def', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name ='')
	
	# restore all the variables needed to calculate the function f
	W = sess.graph.get_tensor_by_name('coef:0')
	b = sess.graph.get_tensor_by_name('bias:0')
	# restore the function
	Y = sess.graph.get_tensor_by_name('Y:0')
	
	saver = tf.train.Saver({'coef':W, 'bias':b})
	saver.restore(sess,'checkpoint-data')
	
	# remember that we need X to calculate f.
	print float(sess.run(Y, {'X:0':np.array([3,2,1]).reshape(-1,3)}))