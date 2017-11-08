import os
import tensorflow as tf
import matplotlib.pyplot as plt
from astropy.wcs.docstrings import name
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x_train = [1, 2, 3]
y_train = [1, 2, 3]


W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = W * x_train + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
