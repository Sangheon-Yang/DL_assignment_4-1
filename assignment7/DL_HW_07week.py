'''
 > Basic MLP Design is from 04WEEK Assignment
 > 'DL_HW_04week.py' is the skeleton code for 'DL_HW_07week.py'
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# ==== Assignment04 =============================

#  784 -> 256 -> 128 -> 256 -> 10     (total 4-layer Design)

W1 = tf.Variable(tf.random_uniform([784, 256], -1., 1.))
b1 = tf.Variable(tf.random_uniform([256], -1., 1.))
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform([256, 128], -1., 1.))
b2 = tf.Variable(tf.random_uniform([128], -1., 1.))
L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_uniform([128, 256], -1., 1.))
b3 = tf.Variable(tf.random_uniform([256], -1., 1.))
L3 = tf.sigmoid(tf.matmul(L2, W3) + b3)

W4 = tf.Variable(tf.random_uniform([256, 10], -1., 1.))
b4 = tf.Variable(tf.random_uniform([10], -1., 1.))
logits = tf.matmul(L3, W4) + b4
hypothesis = tf.nn.softmax(logits)

# ================================================
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = logits))
opt = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(cost)
batch_size = 100


#========= Edited for assignment07 =========#
''' checkpoint '''
ckpt_path = './check_point/cp1.ckpt'
#===========================================#


with tf.Session() as sess:

    saver = tf.train.Saver()
    
    #====================== Edited for assignment07 ============================#
    '''for checkpoint restore '''
    '''- ignore it in first exection, use it from second execution for restore'''
    saver.restore(sess, ckpt_path)
    #===========================================================================#

    sess.run(tf.global_variables_initializer())
    for epoch in range(15):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, opt], feed_dict={X:batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        print('Epoch:', '%d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        
    is_correct = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("Accuracy", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    
    #======== Edited for assignment07 =======#
    '''for checkpoint save'''
    saver.save(sess, ckpt_path)
    #========================================#
