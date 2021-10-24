#include <torch/torch.h>
#include <torch/script.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <torch/types.h>
#include <tuple>
#include "model.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
//#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
//#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur


using namespace std;
const int nr_of_classes = 5;
const int input_img_size = 224;

const double lr_at_high_loss = 0.00003;
const double mo_at_high_loss = 0.25;
const double lr_at_low_loss = 0.000001;
const double mo_at_low_loss = 0.025;

const float loss_high_low_threshold = 0.5f;
double lr = lr_at_high_loss;
double momentum = mo_at_high_loss;
double pre_lr = 0.0;
double pre_momentum = 0.0;

double sum_up_1 = 0.0;
double sum_up_2 = 0.0;

// The batch size for training.
const int64_t kTrainBatchSize = 1;
//const int64_t kTrainBatchSize = 10;

// The batch size for testing.
const int64_t kTestBatchSize = 2;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 100;

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
  cv::Mat rectImage;
  frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
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
    
    if(&file_location != NULL){
    printf("file_location = ");
    cout << file_location << endl;
    }
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

            cv::Mat imgSquarRect = makeSquareImg(img, input_img_size, file_location);
      //      cv::imshow("imgSquarRect", imgSquarRect);
      //      cv::waitKey(100);
            // Convert the image and label to a tensor.
            // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
            // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
            // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
            torch::Tensor img_tensor = torch::from_blob(imgSquarRect.data, {imgSquarRect.rows, imgSquarRect.cols, imgSquarRect.channels()}, torch::kF32).clone();
            img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW
            img_tensor = img_tensor.to(torch::kF32).contiguous();
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
  size_t batch_idx = 0;
  int forloop=0;
  int count = 0;
  float mse = 0.; // mean squared error
  for (auto& batch : data_loader) {
   // torch::manual_seed(batch_idx *45346);


    auto data = batch.data.to(torch::kF32).to(device).clone(), 
    targets = batch.target.squeeze().to(torch::kInt64).to(device).clone();
    optimizer.zero_grad();
    //auto output = model->forward(data);
    auto output = model(data);


    auto loss = torch::nll_loss(output, targets);
/*
    cout << "==================" << endl;
    cout << "targets" << endl;
    cout << targets << endl;
    cout << "output" << endl;
    cout << output << endl;
    cout << "==================" << endl;
*/
/*
    torch::Tensor prob = torch::exp(output);
    cout << "output = " << output << endl;
    printf("Probability of being\n\
    class0 = %.2f percent\n\
    class1 = %.2f percent\n\
    class2 = %.2f percent\n\
    class3 = %.2f percent\n\    
    class4 = %.2f percent\n", prob[0][0].item<float>()*100., prob[0][1].item<float>()*100., prob[0][2].item<float>()*100., prob[0][3].item<float>()*100., prob[0][4].item<float>()*100.);
*/
  //  AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    count++;
    mse += loss.template item<float>();
    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          mse/count);
        //  loss.template item<float>());
          cout << endl;
        printf("count = %d\n", count);
            if((mse/count) < loss_high_low_threshold){
                momentum = mo_at_low_loss;
                lr = lr_at_low_loss; 
            }
            else{
                momentum = mo_at_high_loss;
                lr = lr_at_high_loss; 
            }
        if(lr != pre_lr || momentum != pre_momentum){
            printf("Learning rate lr = %f\n", (float)lr);
            printf("Momentum = %f\n", (float)momentum);
        for (auto param_group : optimizer.param_groups()) {
          //# Static cast needed as options() returns OptimizerOptions (base class)
          static_cast<torch::optim::SGDOptions &>(param_group.options()).lr(lr);
          static_cast<torch::optim::SGDOptions &>(param_group.options()).momentum(momentum);
        }
           
        }
        pre_lr = lr;
        pre_momentum = momentum;
    }
  }
}

template <typename DataLoader>
void test(
    int32_t epoch,
    ObscureResNet &model,
    torch::Device device,
    DataLoader &data_loader,
    torch::optim::Optimizer &optimizer,
    size_t dataset_size)
{
  size_t batch_idx = 0;
  int forloop = 0;
  int count = 0;
  float mse = 0.; // mean squared error

  cv::Mat testImg(input_img_size, input_img_size, CV_32FC3);
  torch::Tensor CPUtensor;
  for (auto &batch : data_loader)
  {
    auto data = batch.data.to(torch::kF32).to(device).clone(),
         targets = batch.target.squeeze().to(torch::kInt64).to(device).clone();
    optimizer.zero_grad();
    torch::Tensor CPUtens = data.to(torch::kCPU);

    cout << "=== Make cv::Mat from dataset tensor ====" << endl;
    float *index_ptr_testImg = testImg.ptr<float>(0);
    sum_up_1 = 0.0;
    for (int r = 0; r < testImg.rows; r++)
    {
      for (int c = 0; c < testImg.cols; c++)
      {
        for (int rgb = 0; rgb < testImg.channels(); rgb++)
        {
          //*index_ptr_testImg = CPUtens[kTestBatchSize-1][rgb][r][c].item<float>();
          *index_ptr_testImg = CPUtens[kTestBatchSize - 1][rgb][r][c].item<float>();
          sum_up_1 += (double)*index_ptr_testImg;
          index_ptr_testImg++;
        }
      }
    }
    cv::imshow("dataset img", testImg);
    cv::waitKey(100);
    cout << "=========================================" << endl;

    //auto output = model->forward(data);
    auto output = model->forward(data);
    torch::Tensor prob = torch::exp(output);
    for (int i = 0; i < kTestBatchSize; i++)
    {
      cout << "target ";
      cout << targets[i] << endl;
      printf("Probability of being\n\
    class0 = %.2f percent\n\
    class1 = %.2f percent\n\
    class2 = %.2f percent\n\
    class3 = %.2f percent\n\    
    class4 = %.2f percent\n",
             prob[i][0].item<float>() * 100., prob[i][1].item<float>() * 100., prob[i][2].item<float>() * 100., prob[i][3].item<float>() * 100., prob[i][4].item<float>() * 100.);
    }
  }
}

auto main() -> int {
  torch::manual_seed(1);
  torch::DeviceType device_type;
    /* initialize random seed: */
  srand (time(NULL));

  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);


  ObscureResNet model(nr_of_classes);
  model->to(device);
  cout << "Number of classes = " << nr_of_classes << endl;

  std::string file_names_csv = "./file_names.csv";
  std::string file_names_csv_test = "./test_file.csv";

  auto custom_dataset = CustomDataset(file_names_csv)
                            .map(torch::data::transforms::Stack<>());
  auto test_dataset = CustomDataset(file_names_csv_test)
                            .map(torch::data::transforms::Stack<>());

 
  int64_t test_lable = 1;


  const size_t custom_dataset_size = custom_dataset.size().value();
  printf("custom_dataset_size = ");
  std::cout << custom_dataset_size << std::endl;

  const size_t test_dataset_size = test_dataset.size().value();
  printf("test_dataset_size = ");
  std::cout << test_dataset_size << std::endl;

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(custom_dataset), kTrainBatchSize);

  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
    model->parameters(), torch::optim::SGDOptions(lr).momentum(momentum));

    //   model->parameters(), torch::optim::SGDOptions(0.00003).momentum(0.25));

  std::string file_names_model = "./latest_model.pt";

  // try to open model file to read
  ifstream ifile;
  ifile.open(file_names_model);
  bool do_load_model = false;
  if (ifile)
  {
    cout << "Model file " << file_names_model << " exists" << endl;
    char charictorInput;
    cout << "Do you want to load latest saved model " << file_names_model << " Y/N " << endl;
    cin >> charictorInput;
    if (charictorInput == 'Y' || charictorInput == 'y')
    {
      do_load_model = true;
    }
    else
    {
      do_load_model = false;
    }
  }
  else
  {
    cout << "Model file " << file_names_model << " dosen't exists";
  }
  if(do_load_model == true)
  {
      torch::load(model, file_names_model);
      printf("Load model =============================================\n");
  }
  else
  {
      printf("Skip load model. Traning from scratch\n");
  }

    string model_path = "./model_s.pt";
    printf("Print Model Structure\n");
    cout << model << endl;
  int test_class = 1;
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

    float loss_flt;


    train(epoch, model, device, *train_loader, optimizer, custom_dataset_size);
    printf("Save model =============================================\n");
    torch::save(model, file_names_model);

    auto b = model->conv4->parameters();
 
    std::string test_file;

    test_class = 3;

     
    printf("\n");
    printf("************ Test image ************* \n");

    test(1, model, device, *test_loader, optimizer, test_dataset_size);

       for (int t=0;t<nr_of_classes;t++){
    test_class = t;

    printf("Test class number = %d\n", test_class);

    std::string str1 = "./data/class";
    std::string str2 = "/";
    std::string str3 = ".jpg";

    int rand_img_indx = rand() % (custom_dataset_size/nr_of_classes);
    std::string test_class_str = std::to_string(test_class);
    std::string test_img_str = std::to_string(rand_img_indx);
    test_file = str1 + test_class_str + str2 + test_img_str + str3;
    test_file = str1 + test_class_str + str2 + "617" + str3;

    printf("test file = ");
    cout << test_file << endl;
    cv::Mat img = cv::imread(test_file);

    cv::Mat imgSquarRect = makeSquareImg(img, input_img_size, test_file);
   
    // Convert the image and label to a tensor.
    // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
    // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
    // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
    torch::Tensor img_tensor = torch::from_blob(imgSquarRect.data, {1, imgSquarRect.rows, imgSquarRect.cols, imgSquarRect.channels()}, torch::kF32).clone();
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to BxCxHxW
    torch::Tensor img_tensor2 = img_tensor.clone();
    printf("Evaluate img_tensor and img_tensor\n");
    auto data = img_tensor.to(device).to(torch::kF32).contiguous();
    cout << "data.sizes() = " << data.sizes() << endl;
    torch::Tensor CPUtensor;
    cout << "=== Make cv::Mat from one sample tensor ====" << endl;
    sum_up_2 = 0.0;
    torch::Tensor CPUtens = data.to(torch::kCPU);
    cv::Mat testImg(input_img_size,input_img_size, CV_32FC3);
    float *index_ptr_testImg = testImg.ptr<float>(0);
    for(int r=0;r<testImg.rows;r++){
      for(int c=0;c<testImg.cols;c++){
        for(int rgb=0;rgb<testImg.channels();rgb++)
        {
          *index_ptr_testImg = CPUtens[0][rgb][r][c].item<float>();
          sum_up_2 += (double)*index_ptr_testImg;
          index_ptr_testImg++;
        }
      }
    }
    cv::imshow("Sample", testImg);
    
    cout << "============**************===================" << endl;
   auto log_prob = model(data);
    torch::Tensor prob = torch::exp(log_prob);
    cout << "log_prob = " << log_prob << endl;
for(int i=0;i<1;i++){
    printf("i = %d\n", i);
    printf("Probability of being\n\
    class0 = %.2f percent\n\
    class1 = %.2f percent\n\
    class2 = %.2f percent\n\
    class3 = %.2f percent\n\    
    class4 = %.2f percent\n", prob[i][0].item<float>()*100., prob[i][1].item<float>()*100., prob[i][2].item<float>()*100., prob[i][3].item<float>()*100., prob[i][4].item<float>()*100.);
}
cv::waitKey(1000);
    }
  }
}
