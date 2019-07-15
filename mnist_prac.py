'''下载数据集'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''建立session并给出参数'''
import tensorflow as tf
sess = tf.compat.v1.InteractiveSession()
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''给出交叉熵'''
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(y),
                                              reduction_indices=[1]))
'''开始训练'''
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.compat.v1.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

'''结果预测'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy count: ' + str(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))