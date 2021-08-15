# deeplearning cpp
用c++编写的一个深度学习框架，主要参考[Andrew Ng](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)的机器学习课程以及Google的[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)。

## 主要功能
这是个以学习为目的的项目，因此功能实现比较简单：
- 创建Sequential拓扑结构的深度学习模型
- 简单的全连接层 (keras中的Dense)
- 支持relu, sigmoid, softmax等activation
- 支持regularization，Dropout
- 支持Onehot表示
- 用[Eigen](https://eigen.tuxfamily.org/)做底层的矩阵运算

不支持GPU运算，也没有graph模式运算，所以运行效率无法满足实际应用的需求。



