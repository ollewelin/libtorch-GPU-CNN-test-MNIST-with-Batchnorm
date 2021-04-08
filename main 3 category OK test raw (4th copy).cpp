#include <torch/torch.h>


#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
//#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
//#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur


using namespace std;
const int nr_of_classes = 3;

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 2;

// The batch size for testing.
const int64_t kTestBatchSize = 48;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 100;

struct ObscureResNet : torch::nn::Module {
  ObscureResNet()
      : 
        conv1(torch::nn::Conv2dOptions(3, 10, /*kernel_size=*/5).stride(2)),
        bn1(torch::nn::BatchNorm2d(10)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        bn2(torch::nn::BatchNorm2d(20)),
        fc1(320, 50),
        fc2(50, nr_of_classes) {
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

// Read in the csv file and return file locations and labels as vector of tuples.
auto ReadCsv(std::string& location) -> std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> {

    std::fstream in(location, std::ios::in);
    std::string line;
    std::string name;
    std::string label;
    std::vector<std::tuple<std::string, int64_t>> csv;

    while (getline(in, line))
    {
        std::stringstream s(line);
        getline(s, name, ',');
        getline(s, label, ',');

        csv.push_back(std::make_tuple("./" + name, stoi(label)));
    }

    return csv;
}

cv::Mat makeSquareImg(cv::Mat frame, int outputImageSize, std::string file_location)
{
  cv::Mat rectImage, graymat;
  //cv::cvtColor(frame, graymat,cv::COLOR_BGR2GRAY);
  frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
  // graymat.convertTo(frame, CV_64FC1, 1.0 / 255.0);
  //graymat.convertTo(frame, CV_32FC1, 1.0f / 255.0f);
  
  //Resize input image to fit into a example 224 x 224 dataset images
  //First step scale the input image so the least side h or w will be resized to example 224
  //Next step crop the larger h or w to example 224 pixel as well, crop with images in center
  //bool inp_landscape;
  int x_start = 0;
  int y_start = 0;
  int inputSize = 0;
  //Find out the smallest side hight or width of the input image
  if (frame.rows == 0 || frame.cols == 0)
  {
    //Zero divition protection here.
    printf("Error! Zero divition protection input image rows = %d cols = %d. Exit program.\n", frame.rows, frame.cols);
    printf("file_location = ");
    cout << file_location << endl;

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
  return rectImage;
}

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
        std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;

    public:
        explicit CustomDataset(std::string& file_names_csv)
            // Load csv file with file locations and labels.
            : csv_(ReadCsv(file_names_csv)) {

        };

        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {

            std::string file_location = std::get<0>(csv_[index]);
            int64_t label = std::get<1>(csv_[index]);

            // Load image with OpenCV.
            cv::Mat img = cv::imread(file_location);

            cv::Mat imgSquarRect = makeSquareImg(img, 56, file_location);
       //     cv::imshow("img", img);
       //     cv::imshow("imgSquarRect", imgSquarRect);
       //     cv::waitKey(1000);
            // Convert the image and label to a tensor.
            // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
            // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
            // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
            torch::Tensor img_tensor = torch::from_blob(imgSquarRect.data, {imgSquarRect.rows, imgSquarRect.cols, imgSquarRect.channels()}, torch::kByte).clone();
            img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW
            
            torch::Tensor label_tensor = torch::full({1}, label);

            return {img_tensor, label_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {

            return csv_.size();
        };
};

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
  int forloop=0;
  /*
  torch::Tensor test_target = torch::zeros({1,3}).to(torch::kF32).to(device);
  torch::Tensor test_out = torch::zeros({1,3}).to(torch::kF32).to(device);
  test_target.requires_grad();
  test_out.requires_grad();
*/
  for (auto& batch : data_loader) {
    auto data = batch.data.to(torch::kF32).to(device), 
    targets = batch.target.squeeze().to(torch::kInt64).to(device);
  
/*
    if(targets.shape()>1)
    {
      printf("exit");
      exit(0);
    }
*/

    optimizer.zero_grad();
    auto output = model.forward(data);
    printf("output tensor\n");
    cout << output << endl;

    auto loss = torch::nll_loss(output, targets);
    //auto loss = torch::nll_loss(output, test_target);
    //auto loss = torch::nll_loss(test_out, test_target);

    
  //  printf("Output forloop %d = ", forloop);
  //  forloop++;
  //  std::cout << output << " Target =" << targets << endl;
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

          cout << endl;
          cout << "target = " << endl;
          cout << targets << endl;
          cout << "******" << endl;
          cout << endl;
          cout << "batch_idx = " << endl;
          cout << batch_idx << endl;
          cout << "******" << endl;

 
    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
          cout << endl;
          cout << "target = " << endl;
      //    cout << targets << endl;
          cout << "====" << endl;
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
  printf("debug1 ====================== \n");
  for (const auto& batch : data_loader) {
    printf("debug2 ====================== \n");
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(output, targets, /*weight=*/{}, torch::Reduction::Sum).template item<float>();
    auto pred = output.argmax(1);
    printf("predict = ");
    std::cout << pred << endl;
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
      printf("dataset_size = %d\n ", dataset_size);
      printf("correct = %d\n ", correct);
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
  
 // device_type = torch::kCPU;
  torch::Device device(device_type);

  ObscureResNet model;
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();


  std::string file_names_csv = "./file_names.csv";
  auto custom_dataset = CustomDataset(file_names_csv)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());
  const size_t custom_dataset_size = custom_dataset.size().value();
  printf("custom_dataset_size = ");
  std::cout << custom_dataset_size << std::endl;

  

  printf("dataset\n");
  //std::cout << train_dataset.dataset() << endl;

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(custom_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();

  auto test_custom_dataset = CustomDataset(file_names_csv)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());
  const size_t test_custom_dataset_size = test_custom_dataset.size().value();

  auto test_loader =
      torch::data::make_data_loader(std::move(test_custom_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));


    torch::Tensor tensor = torch::zeros({2,5});
    std::cout << tensor << std::endl;
    std::cout << tensor.sizes() << std::endl;

    //torch::Tensor tensor2 = read_data()

    printf("Print Model Structure\n");
  //  cout << model.conv1 << endl;
  //  cout << model.conv2 << endl;
  //  cout << model.fc1 << endl;
  //  cout << model.fc2 << endl;
    cout << model << endl;
  //  auto a = model.parameters();
  //  printf("Print Model weights conv1 weights kernels\n");
  //  cout << a[0] << endl;
  int test_apple = 1;
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

   // train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    train(epoch, model, device, *train_loader, optimizer, custom_dataset_size);
   // test(model, device, *test_loader, test_custom_dataset_size);
       // Predict the probabilities for the classes.

    std::string test_file1 = "./data/apples/img45.jpg";
    std::string test_file2 = "./data/bananas/img45.jpg";
    //std::string test_file3 = "./data/cat_w_pray_jpg/45.jpg";
    //std::string test_file3 = "./data/bananas/img4.jpg";
    //std::string test_file2 = "./data/cat_w_pray_jpg/45.jpg";
    //std::string test_file3 = "./data/cat_w_pray_jpg/4.jpg";
    std::string test_file3 = "./data/bananas2/img45.jpg";

    std::string test_file = test_file2;

    if(test_apple == 0)
    {
      test_file = test_file1;
    }
    if(test_apple == 1)
    {
      test_file = test_file2;
    }
    if(test_apple == 2)
    {
      test_file = test_file3;
    }

    if(test_apple < nr_of_classes)
    {
      test_apple++;
    }
    else
    {
      test_apple = 0;
    }
 

    printf("\n");
    printf("************ Test image ************* \n");
    printf("test file = ");
    cout << test_file << endl;
    cv::Mat img = cv::imread(test_file);

    cv::Mat imgSquarRect = makeSquareImg(img, 56, test_file);
    // Convert the image and label to a tensor.
    // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
    // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
    // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
    torch::Tensor img_tensor = torch::from_blob(imgSquarRect.data, {1, imgSquarRect.rows, imgSquarRect.cols, imgSquarRect.channels()}, torch::kByte).clone();
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to BxCxHxW
    
  //  model.eval();
   
    auto data = img_tensor.to(torch::kF32).to(device);

    auto log_prob = model.forward(data);

    torch::Tensor prob = torch::exp(log_prob);
    cout << "log_prob = " << log_prob << endl;
    printf("Probability of being\n\
    an apple = %.2f percent\n\
    a banana = %.2f percent\n\
    a banana2 = %.2f percent\n", prob[0][0].item<float>()*100., prob[0][1].item<float>()*100., prob[0][2].item<float>()*100.);
    cv::imshow("img", img);
    cv::imshow("imgSquarRect", imgSquarRect);
    cv::waitKey(3000);

    // test(model, device, *train_loader, custom_dataset_size);

    //  printf("Print Model weights parts of conv1 weights kernels\n");
    //  cout << a[0][0][0] << endl;
  }
}
