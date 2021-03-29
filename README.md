# libtorch-GPU-CNN-test-MNIST-with-Batchnorm
Test add batchnorm layers.

### Without batch norm layers connected

    Train Epoch: 10 [59584/60000] Loss: 0.0165
    Test set: Average loss: 0.0429 | Accuracy: 0.987

### With batch norm layer attached

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
