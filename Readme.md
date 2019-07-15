# Readme

注：该仓库用于储存机器学习的代码文件，如有必要，请对每个添加进来的代码文件进行简要说明。



## 20190715更新

加入四个文件：

`mnist_prac.py`

` autoencoder_denoise.py` 

` mnist_with_hidden.py` 

` CNN_based_mnist.py`

简要说明：

- `mnist_prac.py`：`mnist`数据集识别，只用了一个`softmax`回归，**无隐含层**，代价函数为**交叉熵**。
- ` autoencoder_denoise.py` ：自编码器的实现，编写了一个**自编码器的类**，有一层隐含层，隐含层激活函数`softplus`，代价函数为**平方误差**。
- ` mnist_with_hidden.py` ：在`mnist`的基础上增加了一层隐含层，并使用`dropout`防止**过拟合**，优化器使用`Adagrad`，激活函数使用`ReLU`，提高了**识别准确率**。
- ` CNN_based_mnist.py`：使用简单的CNN网络识别`mnist`数据集，分别使用两层**卷积层**和**池化层**进行特征的提取，最后使用**全连接层**和一个`dropout`层，激活函数为`softmax`，能够将`mnist`数据集识别准确率提高到99.2%左右。