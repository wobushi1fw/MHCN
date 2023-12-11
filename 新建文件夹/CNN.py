import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST", one_hot=True)
batch_size = 100
n_batchs = mnist.train.num_examples // batch_size
def weight_variable(shape):
    # 权重初始化为服从高斯分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biases_vriable(shape):
    # 偏置初始化为常数值 0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)

x_images = tf.reshape(x, shape=[-1, 28, 28, 1])
# 第一层卷积层
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = biases_vriable([32])
h_conv1 = tf.nn.relu(conv2d(x_images, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = biases_vriable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = biases_vriable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, w_fc1.shape[0]])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = weight_variable([1024, 10])
b_fc2 = biases_vriable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(n_batchs):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, {x: batch_xs, y: batch_ys, keep_prob: 0.7})
        acc, l = sess.run([accuracy, loss], {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter: " + str(epoch) + " Accuracy: " + str(acc) + " Loss: " + str(l))
