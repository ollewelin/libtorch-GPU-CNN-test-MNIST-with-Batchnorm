#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
using namespace std;

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

struct Net : torch::nn::Module {
  Net()
      : 
        conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        bn1(torch::nn::BatchNorm2d(10)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        bn2(torch::nn::BatchNorm2d(20)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
   // register_module("conv3", conv3);
    
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }
    torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::batch_norm(bn1->forward(x), bn1W,bnBias1W,bnmean1W,bnvar1W,true,0.9,0.001,true);
    x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
    x = torch::batch_norm(bn2->forward(x), bn2W,bnBias2W,bnmean2W,bnvar2W,true,0.9,0.001,true);
    x = x.view({-1, 320});
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

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;

//  torch::nn::Conv2d conv3;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

template <typename DataLoader>
void train(
    int32_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(output, targets, /*weight=*/{}, torch::Reduction::Sum).template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));


    torch::Tensor tensor = torch::zeros({2,5});
    std::cout << tensor << std::endl;
    std::cout << tensor.sizes() << std::endl;

    printf("Print Model Structure\n");
    cout << model.conv1 << endl;
    cout << model.conv2 << endl;
    cout << model.fc1 << endl;
    cout << model.fc2 << endl;
    cout << model << endl;
    auto a = model.parameters();
    printf("Print Model weights conv1 weights kernels\n");
    cout << a[0] << endl;

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
 
    
    printf("Print Model weights parts of conv1 weights kernels\n");
    cout << a[0][0][0] << endl;
  }
}
