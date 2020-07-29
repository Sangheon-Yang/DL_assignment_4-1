'''SKELETON CODE for HW 08'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
learning_rate = 0.001
traininf_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X,[-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable(name="W1", shape = [3, 3, 1, 32], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="b1", shape = [32], initializer = tf.contrib.layers.xavier_initializer())
C1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.relu(tf.nn.bias_add(C1, b1))
L1_pool = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')

W2 = tf.get_variable(name="W2", shape = [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="b2", shape = [64], initializer = tf.contrib.layers.xavier_initializer())
C2 = tf.nn.conv2d(L1_pool, W2, strides=[1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(tf.nn.bias_add(C2, b2))
L2_pool = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2_pool, [-1, 64*7*7])

W3 = tf.get_variable(name="W3", shape = [64*7*7, 10], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="b3", shape = [10], initializer = tf.contrib.layers.xavier_initializer())
logits = tf.nn.bias_add(tf.matmul(L2_flat, W3), b3)
hypothesis = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(15):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        print('Epoch:', '%d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        
    is_correct = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("Accuracy", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    
