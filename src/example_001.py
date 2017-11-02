import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

hello = tf.constant("Hello, Tensorflow")
sess = tf.Session()
print(sess.run(hello))