## Github项目推荐 | 深度学习资源，包括一系列架构、模型与建议

AI研习社 [AI研习社](javascript:void(0);) *前天*

![img](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibRiboYcgtAAFwZvvLPUlRkFmiaQ8aCfWBsYib2ic7uVBLAHBtL8m8gYWxDLRdVWaAoASYXjjYclph6NlQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**项目地址：****https://github.com/rasbt/deeplearning-models** 

Jupyter笔记本中TensorFlow和PyTorch的各种深度学习架构，模型和技巧的集合。

## **传统机器学习**

- 感知机 Perceptron [TensorFlow 1] [PyTorch]
- 逻辑回归 Logistic Regression [TensorFlow 1] [PyTorch]
- Softmax回归（多项逻辑回归） Softmax Regression (Multinomial Logistic Regression) [TensorFlow 1] [PyTorch]

## **多层感知机**

- Multilayer Perceptron [TensorFlow 1] [PyTorch]
- Multilayer Perceptron with Dropout [TensorFlow 1] [PyTorch]
- Multilayer Perceptron with Batch Normalization [TensorFlow 1] [PyTorch]
- Multilayer Perceptron with Backpropagation from Scratch [TensorFlow 1] [PyTorch]

## **卷积神经网络**

#### 基本

- Convolutional Neural Network [TensorFlow 1] [PyTorch]
- Convolutional Neural Network with He Initialization [PyTorch]

#### 概念

- Replacing Fully-Connnected by Equivalent Convolutional Layers [PyTorch]

#### 完全卷积

- Fully Convolutional Neural Network [PyTorch]

#### AlexNet

- AlexNet on CIFAR-10 [PyTorch]

#### VGG

- Convolutional Neural Network VGG-16 [TensorFlow 1] [PyTorch]
- VGG-16 Gender Classifier Trained on CelebA [PyTorch]
- Convolutional Neural Network VGG-19 [PyTorch]

#### ResNet

- ResNet and Residual Blocks [PyTorch]
- ResNet-18 Digit Classifier Trained on MNIST [PyTorch]
- ResNet-18 Gender Classifier Trained on CelebA [PyTorch]
- ResNet-34 Digit Classifier Trained on MNIST [PyTorch]
- ResNet-34 Gender Classifier Trained on CelebA [PyTorch]
- ResNet-50 Digit Classifier Trained on MNIST [PyTorch]
- ResNet-50 Gender Classifier Trained on CelebA [PyTorch]
- ResNet-101 Gender Classifier Trained on CelebA [PyTorch]
- ResNet-152 Gender Classifier Trained on CelebA [PyTorch]

#### Network in Network

- Network in Network CIFAR-10 Classifier [PyTorch]

**度量学习**

- Siamese Network with Multilayer Perceptrons [TensorFlow 1]

## **自编码器**

#### 完全连接的自编码器

- Autoencoder [TensorFlow 1] [PyTorch]

#### 卷积自编码器

- Convolutional Autoencoder with Deconvolutions / Transposed Convolutions[TensorFlow 1] [PyTorch]
- Convolutional Autoencoder with Deconvolutions (without pooling operations) [PyTorch]
- Convolutional Autoencoder with Nearest-neighbor Interpolation [TensorFlow 1] [PyTorch]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on CelebA [PyTorch]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on Quickdraw [PyTorch]

#### 变分自编码器

- Variational Autoencoder [PyTorch]
- Convolutional Variational Autoencoder [PyTorch]

#### 条件变分自编码器

- Conditional Variational Autoencoder (with labels in reconstruction loss) [PyTorch]
- Conditional Variational Autoencoder (without labels in reconstruction loss) [PyTorch]
- Convolutional Conditional Variational Autoencoder (with labels in reconstruction loss) [PyTorch]
- Convolutional Conditional Variational Autoencoder (without labels in reconstruction loss) [PyTorch]

## **生成对抗网络（GAN）**

- Fully Connected GAN on MNIST [TensorFlow 1] [PyTorch]
- Convolutional GAN on MNIST [TensorFlow 1] [PyTorch]
- Convolutional GAN on MNIST with Label Smoothing [PyTorch]

## **递归神经网络（RNN）**

#### 多对一：情感分析/分类

- A simple single-layer RNN (IMDB) [PyTorch]
- A simple single-layer RNN with packed sequences to ignore padding characters (IMDB) [PyTorch]
- RNN with LSTM cells (IMDB) [PyTorch]
- RNN with LSTM cells (IMDB) and pre-trained GloVe word vectors [PyTorch]
- RNN with LSTM cells and Own Dataset in CSV Format (IMDB) [PyTorch]
- RNN with GRU cells (IMDB) [PyTorch]
- Multilayer bi-directional RNN (IMDB) [PyTorch]

#### 多对多/序列到序列

- A simple character RNN to generate new text (Charles Dickens) [PyTorch]

## **顺序回归**

- Ordinal Regression CNN -- CORAL w. ResNet34 on AFAD-Lite [PyTorch]
- Ordinal Regression CNN -- Niu et al. 2016 w. ResNet34 on AFAD-Lite [PyTorch]
- Ordinal Regression CNN -- Beckham and Pal 2016 w. ResNet34 on AFAD-Lite [PyTorch]

## **技巧和窍门**

- Cyclical Learning Rate [PyTorch]

## **PyTorch工作流程和机制**

#### 自定义数据集

- Using PyTorch Dataset Loading Utilities for Custom Datasets -- CSV files converted to HDF5 [PyTorch]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Face Images from CelebA [PyTorch]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from Quickdraw [PyTorch]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from the Street View House Number (SVHN) Dataset [PyTorch]

#### 训练和预处理

- Dataloading with Pinned Memory [PyTorch]
- Standardizing Images [PyTorch]
- Image Transformation Examples [PyTorch]
- Char-RNN with Own Text File [PyTorch]
- Sentiment Classification RNN with Own CSV File [PyTorch]

#### 并行计算

- Using Multiple GPUs with DataParallel -- VGG-16 Gender Classifier on CelebA [PyTorch]

#### 其他

- Sequential API and hooks [PyTorch]
- Weight Sharing Within a Layer [PyTorch]
- Plotting Live Training Performance in Jupyter Notebooks with just Matplotlib [PyTorch]

#### Autograd

- Getting Gradients of an Intermediate Variable in PyTorch [PyTorch]

## **TensorFlow工作流程和机制**

#### 自定义数据集

- Chunking an Image Dataset for Minibatch Training using NumPy NPZ Archives [TensorFlow 1]
- Storing an Image Dataset for Minibatch Training using HDF5 [TensorFlow 1]
- Using Input Pipelines to Read Data from TFRecords Files [TensorFlow 1]
- Using Queue Runners to Feed Images Directly from Disk [TensorFlow 1]
- Using TensorFlow's Dataset API [TensorFlow 1]

#### 训练和预处理

- Saving and Loading Trained Models -- from TensorFlow Checkpoint Files and NumPy NPZ Archives [TensorFlow 1]

**![img](https://mmbiz.qpic.cn/mmbiz_gif/bicdMLzImlibRAS3Tao2nfeJk00qqxX3axIgPV3yia4NPESGdUJEM9vsfw1O4Dg1iat7lVNAmbCMY65ia2pzfBXm5kg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)** 点击**阅读原文**，查看本文更多内容



[阅读原文](https://mp.weixin.qq.com/s?__biz=MjM5ODU3OTIyOA==&mid=2650677323&idx=3&sn=0991b3a6f6d34a915f1141bbcc49095f&chksm=bec21d3889b5942ef67472db11380ff5b7526d10bafadb27ecb9d5d4a5218b6116b6198c0488&mpshare=1&scene=1&srcid=0704DpKbkR7eaU2EfGBwc9OT&key=cfad420b0c7e89f93d85e54cb16c23e055bb35db228ce67688c09cb036c20f5505bbc8dfccc0a38225619247034509ca4112e0a7b1197edbec20604598658b3f8f1ce827bb9e8bc81c152ad0291df7f7&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=tFNqUL0VfHxxY99IyVywfi5SR9hyyWsrjaXd5I2BiPMy%2BpgePcB11%2FXQntJivQur##)





![img](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MjM5ODU3OTIyOA==&mid=2650677323&idx=3&sn=0991b3a6f6d34a915f1141bbcc49095f&send_time=)

微信扫一扫
关注该公众号