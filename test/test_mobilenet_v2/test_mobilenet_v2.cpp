#include "core/tensor.h"
#include "graph/graph.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <string>
#include <iostream>

#include <glog/logging.h>
using namespace inferx;

int main()
{
    std::string model_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/pnnx_test/mobile/mobilenet_v2.pnnx.bin";
    std::string param_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/pnnx_test/mobile/mobilenet_v2.pnnx.param";
    std::string dog_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/test_mobilenet_v2/img/dog.jpg";
    graph::Graph graph(model_path, param_path);
    cv::Mat img = cv::imread(dog_path, 1);
    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32FC3);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> channel_imgs;
    cv::split(img, channel_imgs);
    std::cout << channel_imgs.size() << std::endl;
    core::Tensor input_tensor = core::Tensor(core::DataType::DataTypeFloat32, std::vector<uint32_t>{1, 3, 224, 224});
    input_tensor.apply_data();
    // convert to nchw, now is nhwc
    // rgb
    float means[] = {0.485f, 0.456f, 0.406f};
    float vars[] = {0.229f, 0.224f, 0.225f};

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 224; j++)
        {
            for (int k = 0; k < 224; k++)
            {
                input_tensor.ptr<float>()[i * 224 * 224 + j * 224 + k]
                    = (channel_imgs[i].at<float>(j, k) / 255 - means[i]) / vars[i];
            }
        }
    }
    graph.load_model();
    graph.set_input(input_tensor);
    core::Tensor output_tensor;
    auto start = std::chrono::high_resolution_clock::now();
    graph.infernce(output_tensor);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    LOG(INFO) << "Inference time: " << diff.count() * 1000 << " ms";
    int max_id = -1;
    for (uint i = 0; i < output_tensor.size(); i++)
    {
        // LOG(INFO) << "The " << i << "th value is " << output_tensor.ptr<float>()[i];
        if (max_id == -1 || output_tensor.ptr<float>()[i] > output_tensor.ptr<float>()[max_id])
        {
            max_id = i;
        }
    }
    LOG(INFO) << "The max id is " << max_id;
    LOG(INFO) << "The max value is " << output_tensor.ptr<float>()[max_id];
    return 0;

    return 0;
}