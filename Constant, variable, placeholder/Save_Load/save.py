import tensorflow as tf

# define some variables
a = tf.Variable(tf.random_normal([10]), name = 'a')
b = tf.Variable([5,8,3,4,5,6,7,8,9,10], name = 'b', dtype = tf.float32)
c = tf.add(a,b, name = 'c')
init = tf.initialize_all_variables()

with tf.Session() as sess:
    # remember to init before saving data.
    sess.run(init)

    # You can pass a dict to specify the variables you want to save. Otherwise, tensorflow will save all variables in the session.
    # For example: tf.train.Saver({'a':a})
    saver = tf.train.Saver()
    saver.save(sess,'checkpoint-data')

