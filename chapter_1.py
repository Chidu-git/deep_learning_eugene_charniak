# single layer nn

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz = 100

# multi layered neural net (relu[img U + bU ]) V + bV
W = tf.Variable(tf.random.normal([784, 10], stddev=.1))
b = tf.Variable(tf.random.normal([10], stddev=.1))

img = tf.compat.v1.placeholder(tf.float32, [batchSz, 784])
ans = tf.compat.v1.placeholder(tf.float32, [batchSz, 10])

prbs = tf.nn.softmax(tf.matmul(img, W) + b)
xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.math.log(prbs),
                                     reduction_indices=[1]))

train = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(xEnt)
numCorrect = tf.equal(tf.argmax(prbs, 1), tf.argmax(ans, 1))
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# ------------------------------------------------------------

for i in range(1000):
    imgs, anss = mnist.train.next_batch(batchSz)
    acc, _ = sess.run([accuracy, train], feed_dict={img: imgs, ans: anss})

    # sanity check to see accuracy go up
    print(acc)

sumAcc = 0
for i in range(1000):
    imgs, anss = mnist.test.next_batch(batchSz)
    sumAcc += sess.run(accuracy, feed_dict={img: imgs, ans: anss})
print("Test Accuracy: %r" % (sumAcc/1000))
