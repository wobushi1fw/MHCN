import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 创建常量矩阵 1 * 2的与2 * 1的

vec1 = tf.constant([[1,2]])
vec2 = tf.constant([[3],[4]])

# 创建乘法操作，将vec1与vec2进行相乘

res = tf.matmul(vec1, vec2)

#创建会话`session`，执行已经定义的操作

with tf.Session() as sess:
    res = sess.run(res)
    print(res)
