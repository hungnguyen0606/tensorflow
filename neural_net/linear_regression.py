import tensorflow as tf
import numpy as np
import json


config = dict()
config['coef'] = np.array([[3],[5],[1]]).tolist()
config['bias'] = 0.5
config['n_features'] = 3
json.dump(config, open('config.json','w'))

def generate_data(fname, n_samples):
	data = dict()
	X = np.random.uniform(-1, 1, size = [n_samples, config['n_features']]) 
	Y = X.dot(config['coef']) + config['bias']
	data['X'] = X.tolist()
	data['Y'] = Y.tolist()
	json.dump(data, open(fname, 'w'))

def load_data(fname):
	data = json.load(open(fname))
	return np.array(data['X']), np.array(data['Y'])

def generate_model():
	X = tf.placeholder(tf.float32, [None, config['n_features']], name = 'input')
	W = tf.Variable(tf.random_normal([config['n_features'], 1]), name = 'coef')
	bias = tf.Variable(2, name = 'bias', dtype = tf.float32)
	Y = tf.add(tf.matmul(X,W), bias, name = 'output')

	return (X, Y)

# uncomment below lines to generate new data.
# generate_data('training-data', 6000)
# generate_data('validation-data', 2000)
# generate_data('test-data', 2000)

with tf.Session() as sess:
	X, Y = generate_model()
	tf.train.write_graph(sess.graph_def, '.', 'graph_def', False)

	Y_ = tf.placeholder(tf.float32, Y.get_shape(), name = 'label')
	mean_square_error = tf.reduce_mean(tf.square(Y-Y_)) 

	# use gradient descent to minimize loss function: mean square error, with learning rate of 0.01
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(mean_square_error)
	
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()

	# load data used to train model
	x_train, y_train = load_data('training-data')
	x_val, y_val = load_data('validation-data')
	x_test, y_test = load_data('test-data')

	# training phase
	sess.run(init)
	for i in range(2000):
		sess.run(train_step, {'input:0': x_train, 'label:0': y_train})
		print sess.run(mean_square_error, {'input:0': x_val, 'label:0': y_val})

	# save all learned weights
	saver.save(sess, 'checkpoint-data')

