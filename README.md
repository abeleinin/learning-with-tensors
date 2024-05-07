# Learning with Tensors

## The Building Blocks

TODO:

- [ ] Overview of Linear Algebra

- [ ] Gradient Descent and Optimizations

- [ ] Demonstrate regression as a learning task

## Introduction to Deep Learning

### Building a simple Neural Network

![Mnist Network Architectures](img/mnist_nets.png)

Net 1 depicts a single layer network which feeds the input layer of size 784 into a size 10 output layer.

Net 2 depicts a 2-layer network with a single hidden layer of size 12, which feeds into the size 10 output layer.

Net 3 depicts a convolutional neural network using local connectivity and [dropout](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf). This network was inspired by LeNet-5 from the paper [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) by Yann LeCun et al.

Code: [torch_nets.py](mnist/torch_nets.py)

TODO:

- [ ] RNN - Implementing a simple Recurrent Neural Network 

## Implementing Papers: Vision Models

### Implement AlexNet 

![AlexNet Network Architectures](img/alexnet_architecture.png)

Paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

Code: [torch_alexnet.py](alexnet/torch_alexnet.py)

- [ ] ResNet - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
