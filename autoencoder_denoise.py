import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from AdditiveGaussianNoiseAutoencoder import *


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def xavier_init(fan_in, fan_out, constant=1):
    '''定义Xavier初始化器，它的特点是根据某一层网络的输入输出节点数量自动调整最合适的分布'''
    '''参数：fan_in：输入节点数量，
    fan_out：输出节点数量'''
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    '''自定义的去噪自编码的类'''
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer, scale=0.1):
        '''变量名称：
        n_input：输入变量数
        n_hidden：隐藏节点数
        transfer_function：隐藏层激活函数，默认为softplus
        optimizer：优化器，默认为Adam
        scale：高斯噪声系数，默认为0.1
        注：这里只有一个隐含层，可以尝试多加几个隐含层
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.train_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        '''接下来定义网络结构
        变量名称：
        x：维度为n_input的placeholder
        隐藏层就是x加噪声乘以权重矩阵再加上偏置b1，结果再用transfer处理
        经过隐藏层之后，输出层需要复原重建，这里不需要激活函数，只需相乘再加偏置即可'''
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input, )),
            self.weights['w1']), self.weights['b1']
        ))
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']), self.weights['b2'])
        '''定义损失函数
        直接使用平方误差作为cost'''
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        '''参数初始化函数
        将w1，b1，w2，b2存入字典中，只有w1需要用前面的xavier_init()函数初始化，其余的初始化为0'''
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                 self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        '''计算损失及执行一步训练'''
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.train_scale})
        return cost

    def calc_total_cost(self, X):
        '''只求损失'''
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.train_scale})

    def transform(self, X):
        '''返回自编码器隐藏层输出结果'''
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                   self.scale: self.train_scale})

    def generate(self, hidden=None):
        '''将隐含层的结果作为输入，通过重建层将提取到的高阶特征复原为原始数据'''
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        '''输入原数据，输出的是复原后的数据'''
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                   self.scale: self.train_scale})

    def getWeights(self):
        '''获取隐含层权重w1'''
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        '''获取隐含层偏置系数b1'''
        return self.sess.run(self.weights['b1'])


def standard_scale(X_train, X_test):
    '''对数据集进行标准化处理，即0均值，标准差为1，必须保证训练、测试数据的Scaler完全一样'''
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    '''随机获取block：选择一个随机数作为block的起始位置，然后顺序取到一个batch size的数据'''
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# 对训练集、测试集进行标准化变换
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 定义常用参数
n_samples = int(mnist.train.num_examples)
train_epochs = 20
batch_size = 128
display_step = 1

# 创建自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

# 开始训练过程
for epoch in range(train_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=",
              "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
