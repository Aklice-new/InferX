#include "core/tensor.h"
#include "graph/graph.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <string>
using namespace inferx;

int main()
{
    std::string model_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/pnnx_test/mobile/mobile_224.pnnx.bin";
    std::string param_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/pnnx_test/mobile/mobile_224.pnnx.param";
    std::string dog_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/test_mobilenet/img/dog.jpg";
    graph::Graph graph(model_path, param_path);
    cv::Mat img = cv::imread(dog_path);
    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32FC3);
    std::vector<cv::Mat> channel_imgs;
    cv::split(img, channel_imgs);

    core::Tensor input_tensor = core::Tensor(core::DataType::DataTypeFloat32, std::vector<uint32_t>{1, 3, 224, 224});
    // convert to nchw, now is nhwc
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 224; j++)
        {
            for (int k = 0; k < 224; k++)
            {
                input_tensor.ptr<float>()[i * 224 * 224 + j * 224 + k] = channel_imgs[i].at<float>(j, k);
            }
        }
    }
    graph.load_model();
    graph.set_input(input_tensor);
    core::Tensor output_tensor;
    graph.infernce(output_tensor);

    return 0;
}