#pragma once

#include <torch/torch.h>
//torch::Device, torch::Device device
struct ObscureResNetImpl : public torch::nn::Module 
{
  ObscureResNetImpl(int number_of_classes) 
  :     conv1(torch::nn::Conv2dOptions(3, 20, /*kernel_size=*/5).stride(2)),
        bn1(torch::nn::BatchNorm2d(20)),
        conv2(torch::nn::Conv2dOptions(20, 30, 5)),
        bn2(torch::nn::BatchNorm2d(30)),
        conv3(torch::nn::Conv2dOptions(30, 30, 5).padding(2)),
        bn3(torch::nn::BatchNorm2d(30)),
        conv4(torch::nn::Conv2dOptions(30, 30, 5).padding(2)),
        bn4(torch::nn::BatchNorm2d(30)),
        fc1(1080, 100),
        fc2(100, number_of_classes) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    register_module("conv4", conv4);
    register_module("bn4", bn4);
    
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }
    torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::batch_norm(bn1->forward(x), bn1W,bnBias1W,bnmean1W,bnvar1W,true,0.9,0.001,true);
    x = torch::relu(conv2->forward(x));
    x = torch::max_pool2d(x, 2);
    torch::Tensor res1(x.clone());  
    x = torch::batch_norm(bn2->forward(x), bn2W,bnBias2W,bnmean2W,bnvar2W,true,0.9,0.001,true);
    x = torch::relu(conv3->forward(x));
    x += res1;
    x = torch::max_pool2d(x, 2);
    torch::Tensor res2(x.clone());
    x = torch::batch_norm(bn3->forward(x), bn3W,bnBias3W,bnmean3W,bnvar3W,true,0.9,0.001,true);
    x = torch::relu(conv4->forward(x));
    x += res2;
    x = torch::max_pool2d(x, 2);
    x = torch::batch_norm(bn4->forward(x), bn4W,bnBias4W,bnmean4W,bnvar4W,true,0.9,0.001,true);
    x = x.view({-1, 1080});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }
  torch::Tensor bn1W;
  torch::Tensor bnBias1W;
  torch::Tensor bnmean1W;
  torch::Tensor bnvar1W;
  torch::nn::BatchNorm2d bn1;
  torch::Tensor bn2W;
  torch::Tensor bnBias2W;
  torch::Tensor bnmean2W;
  torch::Tensor bnvar2W;
  torch::nn::BatchNorm2d bn2;
  torch::Tensor bn3W;
  torch::Tensor bnBias3W;
  torch::Tensor bnmean3W;
  torch::Tensor bnvar3W;
  torch::nn::BatchNorm2d bn3;
  torch::Tensor bn4W;
  torch::Tensor bnBias4W;
  torch::Tensor bnmean4W;
  torch::Tensor bnvar4W;
  torch::nn::BatchNorm2d bn4;

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Conv2d conv3;
  torch::nn::Conv2d conv4;


  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

TORCH_MODULE(ObscureResNet);