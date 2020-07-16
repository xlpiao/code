import tensorflow as tf
import numpy as np

input = np.zeros((8, 8)) + 1
kernel = np.zeros((5,),dtype=np.float32) + 2

i = tf.placeholder(tf.float32, shape=[1, 1, 8])
k = tf.placeholder(tf.float32, shape = [1, 1, 5])

conv = tf.nn.conv1d(i, k, 1,'SAME')

session = tf.Session()
output = session.run(conv, feed_dict={i:input.reshape((1,1,8)), k:kernel.reshape((1,1,5))})
output.reshape((8, 8))
print(output)
