#pragma once

#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)

const double bn_momentum = 0.7;
const double bn_lr = 0.002;
const int nodes_to_fc1 = 2048;
struct ObscureResNetImpl : public torch::nn::Module 
{
  ObscureResNetImpl(int number_of_classes) 
  :    // conv1(torch::nn::Conv2dOptions(3, 20, /*kernel_size=*/5).stride(2)),
        bn0(torch::nn::BatchNorm2d(3)),
        conv1(torch::nn::Conv2dOptions(3, 64, /*kernel_size=*/7).stride(2)),
        
        bn1(torch::nn::BatchNorm2d(64)),
        conv2(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(1)),
        bn2(torch::nn::BatchNorm2d(64)),
        conv3(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(1)),
        bn3(torch::nn::BatchNorm2d(64)),
        conv4(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(1)),
        bn4(torch::nn::BatchNorm2d(64)),
        conv5(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(1)),
        bn5(torch::nn::BatchNorm2d(64)),
        conv6(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(1)),
        bn6(torch::nn::BatchNorm2d(64)),
        conv7(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(1)),
        bn7(torch::nn::BatchNorm2d(64)),

        conv8(torch::nn::Conv2dOptions(64, 128, /*kernel_size=*/3).stride(2)),
        bn8(torch::nn::BatchNorm2d(128)),
        conv9(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
        bn9(torch::nn::BatchNorm2d(128)),
        conv10(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
        bn10(torch::nn::BatchNorm2d(128)),
        conv11(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
        bn11(torch::nn::BatchNorm2d(128)),
        conv12(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
        bn12(torch::nn::BatchNorm2d(128)),
        conv13(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
        bn13(torch::nn::BatchNorm2d(128)),
        conv14(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
        bn14(torch::nn::BatchNorm2d(128)),
        conv15(torch::nn::Conv2dOptions(128, 128, /*kernel_size=*/3).padding(1)),
        bn15(torch::nn::BatchNorm2d(128)),

        conv16(torch::nn::Conv2dOptions(128, 256, /*kernel_size=*/3).stride(2)),
        bn16(torch::nn::BatchNorm2d(256)),
        conv17(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn17(torch::nn::BatchNorm2d(256)),
        conv18(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn18(torch::nn::BatchNorm2d(256)),
        conv19(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn19(torch::nn::BatchNorm2d(256)),
        conv20(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn20(torch::nn::BatchNorm2d(256)),
        conv21(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn21(torch::nn::BatchNorm2d(256)),
        conv22(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn22(torch::nn::BatchNorm2d(256)),
        conv23(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn23(torch::nn::BatchNorm2d(256)),
        conv24(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn24(torch::nn::BatchNorm2d(256)),
        conv25(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn25(torch::nn::BatchNorm2d(256)),
        conv26(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn26(torch::nn::BatchNorm2d(256)),
        conv27(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),
        bn27(torch::nn::BatchNorm2d(256)),

        conv28(torch::nn::Conv2dOptions(256, 512, /*kernel_size=*/3).stride(2)),
        bn28(torch::nn::BatchNorm2d(512)),
        conv29(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(1)),
        bn29(torch::nn::BatchNorm2d(512)),
        conv30(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(1)),
        bn30(torch::nn::BatchNorm2d(512)),
        conv31(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(1)),
        bn31(torch::nn::BatchNorm2d(512)),
        conv32(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(1)),
        bn32(torch::nn::BatchNorm2d(512)),
        conv33(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(1)),
        bn33(torch::nn::BatchNorm2d(512)),

        fc1(nodes_to_fc1, number_of_classes) {

    register_module("bn0", bn0);

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    register_module("conv4", conv4);
    register_module("bn4", bn4);

    register_module("conv5", conv5);
    register_module("bn5", bn5);
    register_module("conv6", conv6);
    register_module("bn6", bn6);
    register_module("conv7", conv7);
    register_module("bn7", bn7);
    register_module("conv8", conv8);
    register_module("bn8", bn8);

    register_module("conv9", conv9);
    register_module("bn9", bn9);
    register_module("conv10", conv10);
    register_module("bn10", bn10);
    register_module("conv11", conv11);
    register_module("bn11", bn11);
    register_module("conv12", conv12);
    register_module("bn12", bn12);

    register_module("conv13", conv13);
    register_module("bn13", bn13);
    register_module("conv14", conv14);
    register_module("bn14", bn14);
    register_module("conv15", conv15);
    register_module("bn15", bn15);
    register_module("conv16", conv16);
    register_module("bn16", bn16);

    register_module("conv17", conv17);
    register_module("bn17", bn17);
    register_module("conv18", conv18);
    register_module("bn18", bn18);
    register_module("conv19", conv19);
    register_module("bn19", bn19);
    register_module("conv20", conv20);
    register_module("bn20", bn20);

    register_module("conv21", conv21);
    register_module("bn21", bn21);
    register_module("conv22", conv22);
    register_module("bn22", bn22);
    register_module("conv23", conv23);
    register_module("bn23", bn23);
    register_module("conv24", conv24);
    register_module("bn24", bn24);

    register_module("conv25", conv25);
    register_module("bn25", bn25);
    register_module("conv26", conv26);
    register_module("bn26", bn26);
    register_module("conv27", conv27);
    register_module("bn27", bn27);
    register_module("conv28", conv28);
    register_module("bn28", bn28);

    register_module("conv29", conv29);
    register_module("bn29", bn29);
    register_module("conv30", conv30);
    register_module("bn30", bn30);
    register_module("conv31", conv31);
    register_module("bn31", bn31);
    register_module("conv32", conv32);
    register_module("bn32", bn32);

    register_module("conv33", conv33);
    register_module("bn33", bn33);

    register_module("fc1", fc1);
  }
    torch::Tensor forward(torch::Tensor x) {

    x = torch::batch_norm(bn0->forward(x), bn0W,bnBias0W,bnmean0W,bnvar0W,is_training(),bn_momentum,bn_lr,true);
 /* 
    std::cout << x[0][2][0][0] << std::endl;
    torch::Tensor x_cpu = x.clone();
    cv::Mat img(224,224, CV_32FC3, cv::Scalar(0,0,0.3));
    float *pImgData = (float *)img.data;
     // loop through rows, columns and channels
for (int row = 0; row < img.rows; ++row)
{
    for (int column = 0; column < img.cols; ++column)
    {
        for (int channel = 0; channel < img.channels(); ++channel)
        {
            pImgData[img.channels() * (img.cols * row + column) + channel] = x_cpu[0][channel][row][column]->tå();
        }
    }
}

    cv::imshow("img", img);
    cv::waitKey(1);
    

    // example matrix
Mat img = Mat::zeros(256, 128, CV_32FC3);

// get the pointer (cast to data type of Mat)
float *pImgData = (float *)img.data;

// loop through rows, columns and channels
for (int row = 0; row < img.rows; ++row)
{
    for (int column = 0; column < img.cols; ++column)
    {
        for (int channel = 0; channel < img.channels(); ++channel)
        {
            float value = pImgData[img.channels() * (img.cols * row + column) + channel];
        }
    }
}
*/
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::batch_norm(bn1->forward(x), bn1W,bnBias1W,bnmean1W,bnvar1W,is_training(),bn_momentum,bn_lr,true);

    torch::Tensor res1(x.clone());
    x = conv2->forward(x);
    x = torch::batch_norm(bn2->forward(x), bn2W,bnBias2W,bnmean2W,bnvar2W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv3->forward(x);
    x = torch::batch_norm(bn3->forward(x), bn3W,bnBias3W,bnmean3W,bnvar3W,is_training(),bn_momentum,bn_lr,true);
    x += res1;
    x = torch::relu(x);

    torch::Tensor res2(x.clone());
    x = conv4->forward(x);
    x = torch::batch_norm(bn4->forward(x), bn4W,bnBias4W,bnmean4W,bnvar4W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv5->forward(x);
    x = torch::batch_norm(bn5->forward(x), bn5W,bnBias5W,bnmean5W,bnvar5W,is_training(),bn_momentum,bn_lr,true);
    x += res2;
    x = torch::relu(x);

    torch::Tensor res3(x.clone());
    x = conv6->forward(x);
    x = torch::batch_norm(bn6->forward(x), bn6W,bnBias6W,bnmean6W,bnvar6W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv7->forward(x);
    x = torch::batch_norm(bn7->forward(x), bn7W,bnBias7W,bnmean7W,bnvar7W,is_training(),bn_momentum,bn_lr,true);
    x += res3;
    x = torch::relu(x);

    //torch::Tensor res...(x.clone());
    x = conv8->forward(x);
    x = torch::batch_norm(bn8->forward(x), bn8W,bnBias8W,bnmean8W,bnvar8W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv9->forward(x);
    x = torch::batch_norm(bn9->forward(x), bn9W,bnBias9W,bnmean9W,bnvar9W,is_training(),bn_momentum,bn_lr,true);
    //x += res...;
    x = torch::relu(x);

    torch::Tensor res4(x.clone());
    x = conv10->forward(x);
    x = torch::batch_norm(bn10->forward(x), bn10W,bnBias10W,bnmean10W,bnvar10W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv11->forward(x);
    x = torch::batch_norm(bn11->forward(x), bn11W,bnBias11W,bnmean11W,bnvar11W,is_training(),bn_momentum,bn_lr,true);
    x += res4;
    x = torch::relu(x);

    torch::Tensor res5(x.clone());
    x = conv12->forward(x);
    x = torch::batch_norm(bn12->forward(x), bn12W,bnBias12W,bnmean12W,bnvar12W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv13->forward(x);
    x = torch::batch_norm(bn13->forward(x), bn13W,bnBias13W,bnmean13W,bnvar13W,is_training(),bn_momentum,bn_lr,true);
    x += res5;
    x = torch::relu(x);

    torch::Tensor res6(x.clone());
    x = conv14->forward(x);
    x = torch::batch_norm(bn14->forward(x), bn14W,bnBias14W,bnmean14W,bnvar14W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv15->forward(x);
    x = torch::batch_norm(bn15->forward(x), bn15W,bnBias15W,bnmean15W,bnvar15W,is_training(),bn_momentum,bn_lr,true);
    x += res6;
    x = torch::relu(x);

    //torch::Tensor res...(x.clone());
    x = conv16->forward(x);
    x = torch::batch_norm(bn16->forward(x), bn16W,bnBias16W,bnmean16W,bnvar16W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv17->forward(x);
    x = torch::batch_norm(bn17->forward(x), bn17W,bnBias17W,bnmean17W,bnvar17W,is_training(),bn_momentum,bn_lr,true);
    //x += res...;
    x = torch::relu(x);

    torch::Tensor res7(x.clone());
    x = conv18->forward(x);
    x = torch::batch_norm(bn18->forward(x), bn18W,bnBias18W,bnmean18W,bnvar18W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv19->forward(x);
    x = torch::batch_norm(bn19->forward(x), bn19W,bnBias19W,bnmean19W,bnvar19W,is_training(),bn_momentum,bn_lr,true);
    x += res7;
    x = torch::relu(x);

    torch::Tensor res8(x.clone());
    x = conv20->forward(x);
    x = torch::batch_norm(bn20->forward(x), bn20W,bnBias20W,bnmean20W,bnvar20W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv21->forward(x);
    x = torch::batch_norm(bn21->forward(x), bn21W,bnBias21W,bnmean21W,bnvar21W,is_training(),bn_momentum,bn_lr,true);
    x += res8;
    x = torch::relu(x);

    torch::Tensor res9(x.clone());
    x = conv22->forward(x);
    x = torch::batch_norm(bn22->forward(x), bn22W,bnBias22W,bnmean22W,bnvar22W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv23->forward(x);
    x = torch::batch_norm(bn23->forward(x), bn23W,bnBias23W,bnmean23W,bnvar23W,is_training(),bn_momentum,bn_lr,true);
    x += res9;
    x = torch::relu(x);

    torch::Tensor res10(x.clone());
    x = conv24->forward(x);
    x = torch::batch_norm(bn24->forward(x), bn24W,bnBias24W,bnmean24W,bnvar24W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv25->forward(x);
    x = torch::batch_norm(bn25->forward(x), bn25W,bnBias25W,bnmean25W,bnvar25W,is_training(),bn_momentum,bn_lr,true);
    x += res10;
    x = torch::relu(x);

    torch::Tensor res11(x.clone());
    x = conv26->forward(x);
    x = torch::batch_norm(bn26->forward(x), bn26W,bnBias26W,bnmean26W,bnvar26W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv27->forward(x);
    x = torch::batch_norm(bn27->forward(x), bn27W,bnBias27W,bnmean27W,bnvar27W,is_training(),bn_momentum,bn_lr,true);
    x += res11;
    x = torch::relu(x);

    //torch::Tensor res...(x.clone());
    x = conv28->forward(x);
    x = torch::batch_norm(bn28->forward(x), bn28W,bnBias28W,bnmean28W,bnvar28W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv29->forward(x);
    x = torch::batch_norm(bn29->forward(x), bn29W,bnBias29W,bnmean29W,bnvar29W,is_training(),bn_momentum,bn_lr,true);
    //x += res...;
    x = torch::relu(x);

    torch::Tensor res12(x.clone());
    x = conv30->forward(x);
    x = torch::batch_norm(bn30->forward(x), bn30W,bnBias30W,bnmean30W,bnvar30W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv31->forward(x);
    x = torch::batch_norm(bn31->forward(x), bn31W,bnBias31W,bnmean31W,bnvar31W,is_training(),bn_momentum,bn_lr,true);
    x += res12;
    x = torch::relu(x);

    torch::Tensor res13(x.clone());
    x = conv32->forward(x);
    x = torch::batch_norm(bn32->forward(x), bn32W,bnBias32W,bnmean32W,bnvar32W,is_training(),bn_momentum,bn_lr,true);
    x = torch::relu(x);
    x = conv33->forward(x);
    x = torch::batch_norm(bn33->forward(x), bn33W,bnBias33W,bnmean33W,bnvar33W,is_training(),bn_momentum,bn_lr,true);
    x += res13;
    x = torch::relu(x);

    x = torch::relu(torch::avg_pool2d(bn33->forward(x), 2));

    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
  //x = torch::dropout(x, 0.5, true);

  //  std::cout << "print x end" << std::endl;
  //  std::cout << x << std::endl;
  
  //  std::cout << "print x debug6" << std::endl;
  //  std::cout << x[0][0][0] << std::endl;

    x = x.view({-1, nodes_to_fc1});
    x = torch::relu(fc1->forward(x));
    return torch::log_softmax(x, /*dim=*/1);
  }
  
  torch::Tensor bn0W;
  torch::Tensor bnBias0W;
  torch::Tensor bnmean0W;
  torch::Tensor bnvar0W;
  torch::nn::BatchNorm2d bn0;

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

  torch::Tensor bn5W;
  torch::Tensor bnBias5W;
  torch::Tensor bnmean5W;
  torch::Tensor bnvar5W;
  torch::nn::BatchNorm2d bn5;
  torch::Tensor bn6W;
  torch::Tensor bnBias6W;
  torch::Tensor bnmean6W;
  torch::Tensor bnvar6W;
  torch::nn::BatchNorm2d bn6;
  torch::Tensor bn7W;
  torch::Tensor bnBias7W;
  torch::Tensor bnmean7W;
  torch::Tensor bnvar7W;
  torch::nn::BatchNorm2d bn7;
  torch::Tensor bn8W;
  torch::Tensor bnBias8W;
  torch::Tensor bnmean8W;
  torch::Tensor bnvar8W;
  torch::nn::BatchNorm2d bn8;

  torch::nn::Conv2d conv5;
  torch::nn::Conv2d conv6;
  torch::nn::Conv2d conv7;
  torch::nn::Conv2d conv8;

  torch::Tensor bn9W;
  torch::Tensor bnBias9W;
  torch::Tensor bnmean9W;
  torch::Tensor bnvar9W;
  torch::nn::BatchNorm2d bn9;
  torch::Tensor bn10W;
  torch::Tensor bnBias10W;
  torch::Tensor bnmean10W;
  torch::Tensor bnvar10W;
  torch::nn::BatchNorm2d bn10;
  torch::Tensor bn11W;
  torch::Tensor bnBias11W;
  torch::Tensor bnmean11W;
  torch::Tensor bnvar11W;
  torch::nn::BatchNorm2d bn11;
  torch::Tensor bn12W;
  torch::Tensor bnBias12W;
  torch::Tensor bnmean12W;
  torch::Tensor bnvar12W;
  torch::nn::BatchNorm2d bn12;

  torch::nn::Conv2d conv9;
  torch::nn::Conv2d conv10;
  torch::nn::Conv2d conv11;
  torch::nn::Conv2d conv12;

  torch::Tensor bn13W;
  torch::Tensor bnBias13W;
  torch::Tensor bnmean13W;
  torch::Tensor bnvar13W;
  torch::nn::BatchNorm2d bn13;
  torch::Tensor bn14W;
  torch::Tensor bnBias14W;
  torch::Tensor bnmean14W;
  torch::Tensor bnvar14W;
  torch::nn::BatchNorm2d bn14;
  torch::Tensor bn15W;
  torch::Tensor bnBias15W;
  torch::Tensor bnmean15W;
  torch::Tensor bnvar15W;
  torch::nn::BatchNorm2d bn15;
  torch::Tensor bn16W;
  torch::Tensor bnBias16W;
  torch::Tensor bnmean16W;
  torch::Tensor bnvar16W;
  torch::nn::BatchNorm2d bn16;

  torch::nn::Conv2d conv13;
  torch::nn::Conv2d conv14;
  torch::nn::Conv2d conv15;
  torch::nn::Conv2d conv16;

  torch::Tensor bn17W;
  torch::Tensor bnBias17W;
  torch::Tensor bnmean17W;
  torch::Tensor bnvar17W;
  torch::nn::BatchNorm2d bn17;
  torch::Tensor bn18W;
  torch::Tensor bnBias18W;
  torch::Tensor bnmean18W;
  torch::Tensor bnvar18W;
  torch::nn::BatchNorm2d bn18;
  torch::Tensor bn19W;
  torch::Tensor bnBias19W;
  torch::Tensor bnmean19W;
  torch::Tensor bnvar19W;
  torch::nn::BatchNorm2d bn19;
  torch::Tensor bn20W;
  torch::Tensor bnBias20W;
  torch::Tensor bnmean20W;
  torch::Tensor bnvar20W;
  torch::nn::BatchNorm2d bn20;

  torch::nn::Conv2d conv17;
  torch::nn::Conv2d conv18;
  torch::nn::Conv2d conv19;
  torch::nn::Conv2d conv20;

  torch::Tensor bn21W;
  torch::Tensor bnBias21W;
  torch::Tensor bnmean21W;
  torch::Tensor bnvar21W;
  torch::nn::BatchNorm2d bn21;
  torch::Tensor bn22W;
  torch::Tensor bnBias22W;
  torch::Tensor bnmean22W;
  torch::Tensor bnvar22W;
  torch::nn::BatchNorm2d bn22;
  torch::Tensor bn23W;
  torch::Tensor bnBias23W;
  torch::Tensor bnmean23W;
  torch::Tensor bnvar23W;
  torch::nn::BatchNorm2d bn23;
  torch::Tensor bn24W;
  torch::Tensor bnBias24W;
  torch::Tensor bnmean24W;
  torch::Tensor bnvar24W;
  torch::nn::BatchNorm2d bn24;

  torch::nn::Conv2d conv21;
  torch::nn::Conv2d conv22;
  torch::nn::Conv2d conv23;
  torch::nn::Conv2d conv24;

  torch::Tensor bn25W;
  torch::Tensor bnBias25W;
  torch::Tensor bnmean25W;
  torch::Tensor bnvar25W;
  torch::nn::BatchNorm2d bn25;
  torch::Tensor bn26W;
  torch::Tensor bnBias26W;
  torch::Tensor bnmean26W;
  torch::Tensor bnvar26W;
  torch::nn::BatchNorm2d bn26;
  torch::Tensor bn27W;
  torch::Tensor bnBias27W;
  torch::Tensor bnmean27W;
  torch::Tensor bnvar27W;
  torch::nn::BatchNorm2d bn27;
  torch::Tensor bn28W;
  torch::Tensor bnBias28W;
  torch::Tensor bnmean28W;
  torch::Tensor bnvar28W;
  torch::nn::BatchNorm2d bn28;

  torch::nn::Conv2d conv25;
  torch::nn::Conv2d conv26;
  torch::nn::Conv2d conv27;
  torch::nn::Conv2d conv28;

  torch::Tensor bn29W;
  torch::Tensor bnBias29W;
  torch::Tensor bnmean29W;
  torch::Tensor bnvar29W;
  torch::nn::BatchNorm2d bn29;
  torch::Tensor bn30W;
  torch::Tensor bnBias30W;
  torch::Tensor bnmean30W;
  torch::Tensor bnvar30W;
  torch::nn::BatchNorm2d bn30;
  torch::Tensor bn31W;
  torch::Tensor bnBias31W;
  torch::Tensor bnmean31W;
  torch::Tensor bnvar31W;
  torch::nn::BatchNorm2d bn31;
  torch::Tensor bn32W;
  torch::Tensor bnBias32W;
  torch::Tensor bnmean32W;
  torch::Tensor bnvar32W;
  torch::nn::BatchNorm2d bn32;

  torch::nn::Conv2d conv29;
  torch::nn::Conv2d conv30;
  torch::nn::Conv2d conv31;
  torch::nn::Conv2d conv32;

  torch::Tensor bn33W;
  torch::Tensor bnBias33W;
  torch::Tensor bnmean33W;
  torch::Tensor bnvar33W;
  torch::nn::BatchNorm2d bn33;

  torch::nn::Conv2d conv33;

  torch::nn::Linear fc1;
};

TORCH_MODULE(ObscureResNet);