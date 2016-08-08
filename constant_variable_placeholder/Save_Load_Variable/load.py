import tensorflow as tf

# To restore the variable's value, you have to define the variable first.
# You have to set dtype (type of variable) correctly. Fortunately, you can set validate_shape = False to have the tensorflow decide the variable's shape you.
a = tf.Variable(initial_value = -1, validate_shape = False, dtype = tf.float32)
b = tf.Variable(initial_value = -1, validate_shape = False, dtype = tf.float32)
with tf.Session() as sess:
    #Here, you may pass the dictionary including the name and place to store the variables you want to restore. This is where the variable's name works. By setting variable's name, you can easily restore the variable you want.
    saver = tf.train.Saver({'a':a})
    saver.restore(sess, 'checkpoint-data')
    print a.eval()
    # Note if the variable is not restored, it must be initialize by using tf.initialize_all_variables() before using.
    # For example:
    sess.run(tf.initialize_all_variables())
    print b.eval()
    
# You can set name of variable a when defining it and remove the dict in the saver. The program still works.
# You may try to restore c, which eventually causes an error "Tensor name "c" not found in checkpoint files checkpoint-data" because 'c' is not a variable (tensor) but an ops (operation)