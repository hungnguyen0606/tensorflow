import tensorflow as tf
#let define some constants and variables here

#Constant
a = tf.constant(3)
#--------------------------------------------
b = tf.Variable(3.0, name = 'b')
clone_b = tf.Variable(b.initialized_value())
clone2_b = tf.Variable(a)
# Uncomment line below and observe what happens. The interpreter will say that, "Attempting to use uninitialized value b". 
# You should remember to initialize a variable with constant. Otherwise, you have to use initialized_value() to initialize a variable by other variable.
# clone_b = tf.Variable(b)
#--------------------------------------------
c = tf.Variable(tf.random_uniform([10,3]), name = 'c')
#--------------------------------------------
x = tf.placeholder(tf.float32, name = 'input')
#--------------------------------------------
d = tf.add(b, x, name = 'd')
#--------------------------------------------
init = tf.initialize_all_variables()

#start a new session
with tf.Session() as sess:
    #you should call this method before using any variable
    sess.run(init)
    print 'a:', a.eval()
    #--------------------------------------------
    print 'b:', b.eval()
    print 'clone_b:', clone_b.eval()
    print 'clone2_b:', clone2_b.eval()
    #--------------------------------------------
    print 'c:', c.eval()
    #--------------------------------------------
    print 'd:', sess.run(d, {'input:0':3})


