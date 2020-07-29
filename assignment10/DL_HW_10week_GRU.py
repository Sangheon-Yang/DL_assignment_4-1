import tensorflow as tf
import numpy as np

tf. set_random_seed(777)  #reproducibility

sample = "if you want to build a ship, don’t drum up people together to collect wood and don’t assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
idx2char = list(set(sample))  #index -> char
char2idx = {c: i for i, c in enumerate(idx2char)} #char -> index

#hyper Parameters
dic_size = len(char2idx) #RNN input size (one hot size)
hidden_size = 15 #RNN output size
num_classes = len(char2idx) #final output size (RNN or softmax, etc)
batch_size = 1  #one simple data, one batch
sequence_length = len(sample) - 1 #number of lstm rollings( unit #)
learning_rate = 0.1
epoch = 50

sample_idx = [char2idx[c] for c in sample]  #char to index
x_data = [sample_idx[:-1]] # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]] #Y lable sample (1 ~ n) hello: ello

#=====================HW======================#

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

#cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size)
#cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell = tf.contrib.rnn.GRUCell(num_units = hidden_size)

outputs, _state = tf.nn.dynamic_rnn(cell, X_one_hot , dtype = tf.float32)

#==============================================#

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

#reshape out for sequence_loss
outputs = tf. reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = Y, weights = weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session( ) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        l, _ = sess. run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict = {X: x_data})
    
        #print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
    
        print(i, "loss:", l, "Prediction:", ''.join(result_str))
