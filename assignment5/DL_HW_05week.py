import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# for dropout_probability
keep_prob = tf.placeholder(tf.float32)

#  784 -> 256 -> 128 -> 256 -> 10     (total 4-layer Design)

W1 = tf.Variable(tf.random_uniform([784, 256], -1., 1.))
b1 = tf.Variable(tf.random_uniform([256], -1., 1.))
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

#for dropout L1
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_uniform([256, 128], -1., 1.))
b2 = tf.Variable(tf.random_uniform([128], -1., 1.))
L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)

#for dropout L2
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_uniform([128, 256], -1., 1.))
b3 = tf.Variable(tf.random_uniform([256], -1., 1.))
L3 = tf.sigmoid(tf.matmul(L2, W3) + b3)

#for dropout L3
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_uniform([256, 10], -1., 1.))
b4 = tf.Variable(tf.random_uniform([10], -1., 1.))
logits = tf.matmul(L3, W4) + b4

#for dropout Output Layer
logits = tf.nn.dropout(logits, keep_prob)

hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = logits))

# ===== GradientDescentOptimizer in HW 4 =====
#opt = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(cost)

# ===== ADAM_Optimizer for HW 5 =====
opt = tf.train.AdamOptimizer( learning_rate=0.9, beta1=0.6, beta2=0.8, epsilon = 0.1, use_locking=False, name = 'Adam' ).minimize(cost)

batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(15):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, opt], feed_dict={X:batch_xs, Y: batch_ys, keep_prob: 0.7})  # dropout_probability is edited 
            avg_cost += c / total_batch
        print('Epoch:', '%d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        
    is_correct = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("Accuracy", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1 }))



