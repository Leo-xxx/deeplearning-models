## 6000星人气深度学习资源！架构模型技巧全都有，图灵奖得主LeCun推荐

关注前沿科技 [量子位](javascript:void(0);) *今天*

##### 铜灵 发自 凹非寺 量子位 出品 | 公众号 QbitAI

暑假即将到来，不用来充电学习岂不是亏大了。

有这么一份干货，汇集了机器学习**架构**和**模型**的经典知识点，还有各种**TensorFlow**和**PyTorch**的Jupyter Notebook笔记资源，地址都在，无需等待即可取用。

除了取用方便，这份名为**Deep Learning Models**的资源还**尤其全面**。

针对每个细分知识点的介绍还尤其全面的，比如在卷积神经网络部分，作者就由浅及深分别介绍了AlexNet、VGG、ResNet等。

干货发布后，在GitHub短时间获得了**6000+颗星星**，迅速聚集起大量人气。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBavxBn5RfRYVsmibC6dogCQ0TkK6uXznHOyd54tCm0FQKEE4evkw8lCN57VzvFJtwff1fIRKzqsGA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图灵奖得主、AI大牛**Yann LeCun也强烈推荐**，夸赞其为一份不错的PyTorch和TensorFlow Jupyter笔记本推荐！

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBavxBn5RfRYVsmibC6dogCQuaSmCSwn6hqFib8RrI1xYXxPwOo5xTiaLVNTfTWwxLgVJRne4iaiaUP3Sw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这份资源的作者来头也不小，他是威斯康星大学麦迪逊分校的助理教授Sebastian Raschka，此前还编写过Python Machine Learning一书。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtBavxBn5RfRYVsmibC6dogCQZTwDItUvZV9DBb6VZDic7ELbKR01PTrt27fX5ribch1yXS76pEibd5A5w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

话不多说现在进入干货时间，好东西太多篇幅较长，记得**先码后看**！

原资源地址：
https://github.com/rasbt/deeplearning-models

## 干货来也

**1、多层感知机**

多层感知机简称MLP，是一个打基础的知识点：

多层感知机：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-basic.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-basic.ipynb

增加了Dropout部分的多层感知机：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-dropout.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-dropout.ipynb

具备批标准化的多层感知机：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-batchnorm.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-batchnorm.ipynb

从零开始了解多层感知机与反向传播：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-lowlevel.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-fromscratch__sigmoid-mse.ipynb

**2、卷积神经网络**

在卷积神经网络这一部分，细碎的知识点很多，包含基础概念、全卷积网络、AlexNet、VGG等多个内容。来看干货：

卷积神经网络基础入门：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/cnn/cnn-basic.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-basic.ipynb

卷积神经网络的初始化：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-he-init.ipynb

想用等效卷积层替代全连接的话看看下面这个：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/fc-to-conv.ipynb

全卷积神经网络基础知识在这里：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-allconv.ipynb

Alexnet网络模型在CIFAR-10数据集上的实现:

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb

关于VGG模型，你可能需要了解VGG-16架构：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/cnn/cnn-vgg16.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16.ipynb

在CelebA上训练的VGG-16性别分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb

VGG19网络架构：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg19.ipynb

关于2015年被提出的经典CNN模型ResNet，最厉害的资源也在这了。

比如ResNet和残差块：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/resnet-ex-1.ipynb

用MNIST数据集训练的ResNet-18数字分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb

用人脸属性数据集CelebA训练的ResNet-18性别分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb

在MNIST上训练的ResNet-34：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb

在CelebA上训练ResNet-34性别分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-celeba-dataparallel.ipynb

在MNIST上训练的ResNet-50数字分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb

在CelebA上训练ResNet-50性别分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-celeba-dataparallel.ipynb

在CelebA上训练ResNet-101性别分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb

在CelebA上训练ResNet-152性别分类器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet152-celeba.ipynb

CIFAR-10分类器中的网络：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10.ipynb

**3、指标学习**

具有多层感知机的孪生网络：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/metric/siamese-1.ipynb

**4、自编码器**

在自编码器这一部分，同样有很多细分类别需要学习，注意留出充足时间学习这一内容。

自编码器的种类很多，比如全连接自编码器：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-basic.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-basic.ipynb

还有卷积自编码器。比如这个反卷积（转置卷积）卷积自编码器：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-deconv.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv.ipynb

没有进行池化的反卷积自编码器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv-nopool.ipynb

有最近邻插值的卷积自编码器：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-conv-nneighbor.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor.ipynb

在CelebA上训练过的有最近邻插值的卷积自编码器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb

在谷歌涂鸦数据集Quickdraw上训练过的有最近邻插值的卷积自编码器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor-quickdraw-1.ipynb

变分自编码器也是自编码器中的重要一类：

变分自编码器基础介绍：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-var.ipynb

卷积变分自编码器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-var.ipynb

最后，还有条件变分自编码器也需要关注。比如在重建损失中有标签的：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cvae.ipynb

没有标签的：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cvae_no-out-concat.ipynb

有标签的条件变分自编码器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae.ipynb

没有标签的条件变分自编码器：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb

**5、生成对抗网络（GAN）**

在MNIST上的全连接GAN：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/gan.ipynb

在MNIST上训练的条件GAN：

> TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan-conv.ipynb
>
> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/gan-conv.ipynb

用Label Smoothing方法优化过的条件GAN：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/gan-conv-smoothing.ipynb

**6、循环神经网络**

针对多对一的情绪分析和分类问题中，包括简单单层RNN：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_imdb.ipynb

压缩序列的简单单层RNN：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb

RNN和LSTM技术：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_imdb.ipynb

基于GloVe预训练词向量的有LSTM核的RNN：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_imdb-glove.ipynb

GRU核的RNN：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb

多层双向RNN：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb

一对多/序列到序列的生成新文本的字符RNN：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb

**7、有序回归**

针对不同场景，有三类有序回归干货：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb
>
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb
>
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb

**8、方法和技巧**

关于周期性学习速率，这里也有一份小技巧：

> PyTorch版
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/tricks/cyclical-learning-rate.ipynb

**9、PyTorch Workflow和机制**

用自定义数据集加载PyTorch，这里也有一些攻略：

比如用CelebA中的人脸图像：

> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-celeba.ipynb

比如用街景数据集：

> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-svhn.ipynb

比如用Quickdraw:

> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb

在训练和预处理环节，标准化图像可参考：

> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-standardized.ipynb

图像信息样本：

> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/torchvision-transform-examples.ipynb

有文本文档的Char-RNN ：

> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb

在CelebA上训练的VGG-16性别分类器的并行计算等：

> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb

**10、TensorFlow Workflow与机制**

这是这份干货中的最后一个大分类，包含自定义数据集、训练和预处理两大部分。

内容包括：

> 将NumPy NPZ用于小批量训练图像数据集
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/image-data-chunking-npz.ipynb
>
> 用HDF5文件存储图像数据集，用于小规模训练
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/image-data-chunking-hdf5.ipynb
>
> 用输入pipeline从TFRecords文件中读取数据
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/tfrecords.ipynb
>
> TensorFlow数据集API
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/dataset-api.ipynb

如果需要从TensorFlow Checkpoint文件和NumPy NPZ Archive中存储和加载训练模型，可移步：

> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/saving-and-reloading-models.ipynb

**11、传统机器学习**

最后，如果你是从零开始入门，可以从传统机器学习看起。包括感知机、逻辑回归和Softmax回归等。

> 感知机部分TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/perceptron.ipynb
>
> PyTorch版笔记
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/perceptron.ipynb

逻辑回归部分也是一样：

> 逻辑回归部分部分TensorFlow版Jupyter Notebooks
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/logistic-regression.ipynb
>
> PyTorch版笔记
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/logistic-regression.ipynb

Softmax回归，也称为多项逻辑回归：

> Softmax回归部分部分TensorFlow版Jupyter Notebook
> https://github.com/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/softmax-regression.ipynb
>
> PyTorch版笔记
> https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/softmax-regression.ipynb

## 传送门

这份干货满满的资源到这里就结束了，再次放上原文传送门：

https://github.com/rasbt/deeplearning-models

超强干货，记得收藏~

— **完** —

**AI社群 | 与优秀的人交流**

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtBavxBn5RfRYVsmibC6dogCQvv70E942hsKiaqXajAINPb2sGlxLXxqX8tpUWvdg2Y7vxOpL2ryB3gw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**精选直播 | 大牛的观点碰撞**

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtBavxBn5RfRYVsmibC6dogCQHvUMicVQJpFefgRUQuAQSWaJicQapQIp2jr4mh0Me9bTkjnfVq0SHSbw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtBavxBn5RfRYVsmibC6dogCQxd8PPt5FPCVNTOxJibdE0DDyRHULIAMxhLloHv1C4Ljzvz8M8b207Ew/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**量子位** QbitAI · 头条号签约作者





վ'ᴗ' ի 追踪AI技术和产品新动态



喜欢就点「好看」吧！ 











微信扫一扫
关注该公众号