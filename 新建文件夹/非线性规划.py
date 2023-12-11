import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

class xianxingguihua(object):
    def __init__(self):
        # 占位符运算
        self.x = None
        self.noise = None
        self.y_value = None
        self.W_1 = None
        self.b_1 = None
        self.W_2 = None
        self.b_2 = None

        # 输入数据
        self.x_data = None
        self.noise_data = None
        self.y_label = None

        # 创建x与noise占位
        self.x = tf.placeholder(tf.float32, [None, 1], name="x_input")
        self.noise = tf.placeholder(tf.float32, [None, 1], name="y_input")

        # 初始化模型参数
        self.W_1 = tf.Variable(tf.random_normal([1, 10]))
        self.b_1 = tf.Variable(tf.zeros([1, 10]))
        self.W_2 = tf.Variable(tf.random_normal([10, 1]))
        self.b_2 = tf.Variable(tf.zeros([1, 1]))

        # 获取输入数据
        self.input()

    def input(self):
        # 得到输入数据
        self.x_data = np.linspace(-0.5, 0.5, 200)
        self.x_data = self.x_data[:, np.newaxis]
        # 得到输入噪音数据
        self.noise_data = np.random.normal(0, 0.02, self.x_data.shape)
        self.y_label = np.square(self.x_data) + self.noise_data

    # 定义神经网络：两层，隐藏层和输出层
    def model(self):
        a_1 = tf.matmul(self.x, self.W_1) + self.b_1
        out_1 = tf.nn.tanh(a_1)
        a_2 = tf.matmul(out_1, self.W_2) + self.b_2
        out_2 = tf.nn.tanh(a_2)
        return out_2

    # 计算损失函数操作
    def loss(self):
        return tf.reduce_mean(tf.square(self.y_value - self.y_label))

    def trainModel(self):
        self.y_value = self.model()
        l1 = self.loss()
        # 优化损失函数操作
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(l1)
        # 初始化操作
        init = tf.global_variables_initializer()
        # 定义需要获取的变量
        variables_to_fetch = [l1, self.W_1, self.b_1, self.W_2, self.b_2]

        with tf.Session() as sess:
            sess.run(init)
            for epc in range(10000):
                _, loss_value = sess.run([train_step, l1], feed_dict={self.x: self.x_data, self.noise: self.noise_data})
                # 每100次迭代打印一次损失
                if epc % 100 == 0:
                    print(f'Epoch {epc}: Loss = {loss_value}')

            # 获取最终的损失和模型参数
            final_loss, W_1, b_1, W_2, b_2 = sess.run(variables_to_fetch, feed_dict={self.x: self.x_data, self.noise: self.noise_data})

            # 打印学习到的参数
            print(f'Final Loss: {final_loss}')
            print(f'Final W_1: {W_1}')
            print(f'Final b_1: {b_1}')
            print(f'Final W_2: {W_2}')
            print(f'Final b_2: {b_2}')

            # 画出学习的曲线和原始数据
            prediction_value = sess.run(self.y_value, feed_dict={self.x: self.x_data, self.noise: self.noise_data})
            plt.scatter(self.x_data, self.y_label)
            plt.plot(self.x_data, prediction_value, "r-", lw=3)
            plt.show()

# 创建对象并训练模型
ha = xianxingguihua()
ha.trainModel()