# libtorch-GPU-CNN-test-MNIST-with-Batchnorm
Test add batchnorm layers.

Was modifyed this code with adding batchnorm layer between each convolution layers

https://github.com/goldsborough/examples/tree/cpp/cpp/mnist

### Youtube video
https://www.youtube.com/watch?v=wLQbXEORgFA

### MNIST datasets

#### MNIST Fashion dataset

https://github.com/zalandoresearch/fashion-mnist

#### Or use the old MNIST digits dataset

http://yann.lecun.com/exdb/mnist/

### Without batch norm layers connected (on MNIST digits dataset)

    Train Epoch: 10 [59584/60000] Loss: 0.0165
    Test set: Average loss: 0.0429 | Accuracy: 0.987

### With batch norm layer attached (on MNIST digits dataset)

    Train Epoch: 10 [59584/60000] Loss: 0.0120
    Test set: Average loss: 0.0315 | Accuracy: 0.989



Example print out

    Train Epoch: 10 [59584/60000] Loss: 0.0120
    Test set: Average loss: 0.0315 | Accuracy: 0.989
    Print Model weights parts of conv1 weights kernels
    0.0714 -0.0887 -0.2127 -0.1545 -0.0813
    0.1184  0.1395  0.0606  0.0129  0.0564
    -0.0033  0.1634  0.2492  0.1134  0.0322
    -0.0914 -0.0334  0.0359  0.1716  0.1377
    -0.1568 -0.1173 -0.1753 -0.1878 -0.0052
    [ CUDAFloatType{5,5} ]


### Continue exploring Libtorch C++ with OpenCV towards a plane simple ResNet-34 training from scrach with custom image dataset.

The code snippet :

        under construction main.cpp

I will try to do a (mid level programming) of a fix plain ResNet-34 (hardcoded ResNet-34 not generic ResNet-X with bottlenecks etc).
Toghether with custom data set using OpenCV for a classification of color images or video stream. Not need using torchvision for this yet.

#### Prepare dataset tensor from abriarity size of test.jpg input image

![]"Prepare {1, 3, 224, 224} tensor from test_jpg.png"
