import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

# x与标签y
x_data = np.random.rand(100)
y_data = 0.1 * x_data + 0.2

# 斜率和偏置
k = tf.Variable(0.)
b = tf.Variable(0.)

# 优化器创建
optim = tf.train.GradientDescentOptimizer(0.2)

# 计算估值y操作
y = k * x_data + b
# 计算损失操作
loss = tf.reduce_mean(tf.square(y - y_data))
# 优化操作
train_step = optim.minimize(loss)
# 初始化操作
init = tf.global_variables_initializer()

# 创建会话
with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        sess.run(train_step)

    k_value, b_value, y_value, loss_value=sess.run([k, b, y, loss])
    print("k={}, b={}, y={}, loss={}".format(k_value, b_value,y_value ,loss_value))

