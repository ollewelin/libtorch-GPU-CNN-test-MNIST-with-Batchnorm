#include <torch/torch.h>


#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
//#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
//#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur


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

struct ObscureResNet : torch::nn::Module {
  ObscureResNet()
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

// You can for example just read your data and directly store it as tensor.
torch::Tensor read_data()
{
  char filename[100];
  FILE *fp2 = fopen(filename, "r");
  sprintf(filename, "test.jpg"); //Assigne a filename with index number added
  fp2 = fopen(filename, "r");
  if (fp2 == NULL)
  {
    printf("Error while opening file test.jpg");
    exit(0);
  }

  //  torch::Tensor tensor = ....
  cv::Mat frame, rectImage;
  frame = cv::imread(filename, 1);

  fclose(fp2);
  printf("Close file\n");

 // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
  frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
  const int outputImageSize = 224;
  //Resize input image to fit into a 224 x 224 dataset images
  //First step scale the input image so the least side h or w will be resized to 224
  //Next step crop the larger h or w to 224 pixel as well, crop with images in center
  //bool inp_landscape;
  int x_start = 0;
  int y_start = 0;
  int inputSize = 0;
  //Find out the smallest side hight or width of the input image
  if (frame.rows == 0 || frame.cols == 0)
  {
    //Zero divition protection here.
    printf("Error! Zero divition protection input image rows = %d cols = %d. Exit program.\n", frame.rows, frame.cols);
    exit(0);
  }
  else
  {
    if (frame.rows > frame.cols)
    {
      //Input images is a portrait mode
      //Calculate the starting point of the square rectangle
      inputSize = frame.cols;
      y_start = (frame.rows / 2) - inputSize / 2;
      x_start = 0;
    }
    else
    {
      //Input images is a landscape mode
      //Make a square rectangle of input image
      inputSize = frame.rows;
      x_start = (frame.cols / 2) - inputSize / 2;
      y_start = 0;
    }
    //Make a square rectangle of input image
    //Mat rect_part(image, Rect(rand_x_start, rand_y_start, Width, Height));//Pick a small part of image
    cv::Mat rectImageTemp(frame, cv::Rect(x_start, y_start, inputSize, inputSize)); //
    //Size size(input_image_width,input_image_height);//the dst image size,e.g.100x100
    cv::Size outRectSize(outputImageSize, outputImageSize);
    //resize(src,dst,size);//resize image
    cv::resize(rectImageTemp,rectImage,outRectSize);

  }
  
  cv::imshow("frame", frame);
  cv::imshow("rectImage", rectImage);
  
  

//https://discuss.pytorch.org/t/libtorch-c-convert-a-tensor-to-cv-mat-single-channel/47701/5

  int kCHANNELS = 3;
  int rectImage_h = outputImageSize;
  int rectImage_w = outputImageSize;
  auto input_tensor = torch::from_blob(rectImage.data, {1, rectImage_h, rectImage_w, kCHANNELS});
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  std::cout << input_tensor.sizes() << std::endl;
  cv::waitKey(1000);
  return input_tensor;
};

/*
vector<torch::Tensor> process_images(vector<string> list_images)
{
    vector<torch::Tensor> images;
   // images.push_back(10);
};
*/
/* Loads images to tensor type in the string argument */
vector<torch::Tensor> process_images(vector<string> list_images, vector<torch::Tensor> nisse) {
  cout << "Reading Images..." << endl;
  // Return vector of Tensor form of all the images


  
  return nisse ;
}
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
  // Declare 2 vectors of tensors for images and labels
  vector<torch::Tensor> images, labels;
public:
  
  // Constructor
  //CustomDataset(vector<string> list_images, vector<string> list_labels) {
    CustomDataset() {
      images.push_back(read_data());
      images.push_back(read_data());
      images.push_back(read_data());
   // images = process_images(list_images);
   //images = read_data
  //  images = process_images();

  //  labels = process_labels(list_labels);
  };

  // Override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override {
    torch::Tensor sample_img = images.at(index);
    torch::Tensor sample_label = labels.at(index);
    return {sample_img.clone(), sample_label.clone()};
  };

  // Return the length of data
  torch::optional<size_t> size() const override {
    return labels.size();
  };
};

/*
class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        torch::Tensor states_, labels_;

    public:
        
        explicit MyDataset(const std::string& loc_states, const std::string& loc_labels) 
            : states_(read_data(loc_states)),
              labels_(read_data(loc_labels) {   };

        torch::data::Example<> get(size_t index) override;
};
*/
template <typename DataLoader>
void train(
    int32_t epoch,
    ObscureResNet& model,
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
    ObscureResNet& model,
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
  torch::Tensor tensor13 = torch::zeros({3,2});
  torch::Tensor tensor14 = torch::ones({3,4});
 // torch::Tensor tensor = torch::zeros({2,5});
  std::vector<torch::Tensor> tensor15;
  tensor15.push_back(tensor14);
  tensor15.push_back(tensor13);
//  tensor15[0] = tensor14;
 // tensor15[1] = tensor14;
//  std::cout << tensor15[0] << std::endl;
  std::cout << tensor15[0] << std::endl;
  std::cout << tensor15[1] << std::endl;

 // nisse.push_back();
 // nisse.push_back()
 // vector<torch::Tensor> tensor12 = torch::zeros({2,5});
 // nisse.push_back();
  vector<int> blabla;
  //blabla.push_back(3);
  blabla.push_back(9);
 // process_images(vector<string> {"hej", "fff"}, nisse);
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  ObscureResNet model;
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();

  auto custom_dataset = CustomDataset().map(torch::data::transforms::Stack<>());
  const size_t custom_dataset_size = custom_dataset.size().value();
  printf("custom_dataset_size = ");
  std::cout << custom_dataset_size << std::endl;

  printf("dataset\n");
  //std::cout << train_dataset.dataset() << endl;

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

    torch::Tensor tensor2 = read_data();

    printf("Print Model Structure\n");
  //  cout << model.conv1 << endl;
  //  cout << model.conv2 << endl;
  //  cout << model.fc1 << endl;
  //  cout << model.fc2 << endl;
    cout << model << endl;
  //  auto a = model.parameters();
  //  printf("Print Model weights conv1 weights kernels\n");
  //  cout << a[0] << endl;

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
 
    
  //  printf("Print Model weights parts of conv1 weights kernels\n");
  //  cout << a[0][0][0] << endl;
  }
}
