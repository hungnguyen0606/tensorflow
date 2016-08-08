import tensorflow as tf

with tf.Session() as sess:
	with open('graph_def', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		# you should set 'name' to '' instead of removing it, which will cause errors.
		tf.import_graph_def(graph_def, name = '')

	# load model
	W = sess.graph.get_tensor_by_name('coef:0')
	b = sess.graph.get_tensor_by_name('bias:0')
	Y = sess.graph.get_tensor_by_name('output:0')

	# load learned coefficients and bias
	saver = tf.train.Saver({'coef':W, 'bias':b})
	saver.restore(sess, 'checkpoint-data')

	# W, b should be approximately [[3],[5],[1]] and 0.5, correspondingly.
	print W.eval()
	print b.eval()

	#Now, you can reuse the model as following:
	print sess.run(Y, {'input:0':[[3,3,4]]})