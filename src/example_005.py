import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


tf.set_random_seed(777)

x_data = [1,2,3]
y_data = [3,6,9]

W = tf.Variable(tf.random_normal([1], name='weight'))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
