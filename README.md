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

## 配置运行环境
在linux环境下，安装以下模块:
- 安装[CMake](https://cmake.org/)
- 安装[C++ Boost Library](https://www.boost.org/)
- 安装[OpenBLAS](https://www.openblas.net/)

修改deeplearning目录下的[CMakeList.txt](https://github.com/gaolichen/deeplearning-cpp/blob/main/deeplearning/CMakeLists.txt) 文件


```
SET(BOOST_ROOT ${MYDEV}/boost_1_70_0) => SET(BOOST_ROOT /path/to/boost/directory)

SET(BLAS_LIBRARIES /opt/OpenBLAS/lib/libopenblas.so) => SET(BLAS_LIBRARIES /path/to/libopenblas.so)

```

## 如何运行
- 在deeplearning目录下创建build目录
- 在build目录下运行命令行 `cmake ..`和`make`
- 在`build/src`目录下，输入`./dpltest`运行单元测试，输入`./dpltest --run_test=Demo_suite`运行demo任务

运行效果:
![运行效果](https://github.com/gaolichen/deeplearning-cpp/blob/062641cb08dc791df5ef9d46782515a40764aeb9/dpltest.png)
