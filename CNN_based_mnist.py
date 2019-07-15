'''下载数据集'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


def weight_varible(shape):
    '''由于有很多的权重和偏置需要创建，定义初始化函数以便重复使用'''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_varible(shape):
    '''定义偏置'''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    '''
    卷积层需要重复使用，因此创建函数
    :param x: 输入
    :param W: 卷积参数
    :return: 卷积输出结果
    '''
    # strides为卷积模板移动的步长，padding为边界的处理方式，选择SAME让卷积的输入输出保持同样的尺寸
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    '''
    使用2x2的最大池化，也就是将2x2的像素块降低为1x1
    :param x:输入
    :return:池化后的结果
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# 先定义输入参数
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 将原来的图像转化为28x28的形式
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_varible([32])
# 第一层隐含层的输出
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 对隐含层的结果进行池化
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_varible([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 注：经过两次池化之后，图像大小从28x28已经变为现在的7x7

# 接下来对第二个卷积层的输出tensor进行变形，转化为1D的向量，再连接一个全连接层
W_fc1 = weight_varible([7 * 7 * 64, 1024])
b_fc1 = bias_varible([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 加一个Dropout层减轻过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_varible([1024, 10])
b_fc2 = bias_varible([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义损失函数，仍使用交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义评测准确率的操作
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练过程
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels,
                                               keep_prob: 1.0})
        print("step %d, training accuracy %g" %(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
