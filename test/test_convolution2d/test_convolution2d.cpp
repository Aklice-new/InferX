#include "core/tensor.h"
#include "layer/kernels/convolution2d.h"
#include "utils/utils.h"

#include <cstdint>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>

// Input shape :  torch.Size([1, 64, 112, 112])
float input[1 * 64 * 112 * 112];

// weight
float weight1[9];
float weight2[9];
float weight3[25];
float weight4[49];

// bias
float bias1[64];
float bias2[64];
float bias3[64];
float bias4[64];

// Output1 shape :  torch.Size([1, 64, 55, 55])
// Output2 shape :  torch.Size([1, 64, 28, 28])
// Output3 shape :  torch.Size([1, 64, 22, 22])
// Output4 shape :  torch.Size([1, 64, 37, 37])
float output1[1 * 64 * 55 * 55];
float output2[1 * 64 * 28 * 28];
float output3[1 * 64 * 22 * 22];
float output4[1 * 64 * 37 * 37];

using namespace inferx::layer;

TEST(Convolution2dLayer, test1)
{
    // torch.nn.Conv2d(kernel_size=3, stride=2)
    // Output1 shape :  torch.Size([1, 64, 55, 55])

    Convolution2DLayer convolution2d("convolution2d");
    convolution2d.kernel_h_ = 3;
    convolution2d.kernel_w_ = 3;
    convolution2d.stride_h_ = 2;
    convolution2d.stride_w_ = 2;
    convolution2d.padding_h_ = 1;
    convolution2d.padding_w_ = 1;
    convolution2d.use_bias_ = true;
    convolution2d.dilation_h_ = 1;
    convolution2d.dilation_w_ = 1;
    convolution2d.groups_ = 1;
    convolution2d.in_channels_ = 64;
    convolution2d.out_channels_ = 64;

    // weight
    std::vector<uint32_t> weight_shape = {9};
    Tensor::TensorPtr weight = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight->copy_from(weight1, weight->size());
    // bias
    std::vector<uint32_t> bias_shape = {64};
    Tensor::TensorPtr bias = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias->copy_from(bias1, bias->size());

    convolution2d.bias_ = {bias};
    convolution2d.weights_ = {weight};

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, input_tensor->size());
    std::vector<uint32_t> output_shape = {1, 64, 55, 55};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    convolution2d.prepare_layer({input_tensor}, {output_tensor});
    convolution2d.forward_cpu();

    for (int i = 0; i < 1 * 64 * 55 * 55; i++)
    {
        ASSERT_NEAR(output1[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(Convolution2dLayer, test2)
{
    // torch.nn.Conv2d(kernel_size=3, stride=4, padding=1)
    // Output2 shape :  torch.Size([1, 64, 28, 28])

    Convolution2DLayer convolution2d("convolution2d");
    convolution2d.kernel_h_ = 3;
    convolution2d.kernel_w_ = 3;
    convolution2d.stride_h_ = 4;
    convolution2d.stride_w_ = 4;
    convolution2d.padding_h_ = 1;
    convolution2d.padding_w_ = 1;
    convolution2d.use_bias_ = true;
    convolution2d.dilation_h_ = 1;
    convolution2d.dilation_w_ = 1;
    convolution2d.groups_ = 1;
    convolution2d.in_channels_ = 64;
    convolution2d.out_channels_ = 64;

    // weight
    std::vector<uint32_t> weight_shape = {9};
    Tensor::TensorPtr weight = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight->copy_from(weight2, weight->size());
    // bias
    std::vector<uint32_t> bias_shape = {64};
    Tensor::TensorPtr bias = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias->copy_from(bias2, bias->size());

    convolution2d.bias_ = {bias};
    convolution2d.weights_ = {weight};

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 28, 28};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    convolution2d.prepare_layer({input_tensor}, {output_tensor});
    convolution2d.forward_cpu();

    for (int i = 0; i < 1 * 64 * 28 * 28; i++)
    {
        ASSERT_NEAR(output2[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(Convolution2dLayer, test3)
{
    // torch.nn.Conv2d(kernel_size = 5, stride = 5, padding = 1)
    // Output3 shape :  torch.Size([1, 64, 22, 22])
    Convolution2DLayer convolution2d("convolution2d");
    convolution2d.kernel_h_ = 5;
    convolution2d.kernel_w_ = 5;
    convolution2d.stride_h_ = 5;
    convolution2d.stride_w_ = 5;
    convolution2d.padding_h_ = 1;
    convolution2d.padding_w_ = 1;
    convolution2d.use_bias_ = true;
    convolution2d.dilation_h_ = 1;
    convolution2d.dilation_w_ = 1;
    convolution2d.groups_ = 1;
    convolution2d.in_channels_ = 64;
    convolution2d.out_channels_ = 64;

    // weight
    std::vector<uint32_t> weight_shape = {25};
    Tensor::TensorPtr weight = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight->copy_from(weight3, weight->size());
    // bias
    std::vector<uint32_t> bias_shape = {64};
    Tensor::TensorPtr bias = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias->copy_from(bias3, bias->size());

    convolution2d.bias_ = {bias};
    convolution2d.weights_ = {weight};

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 22, 22};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    convolution2d.prepare_layer({input_tensor}, {output_tensor});
    convolution2d.forward_cpu();

    for (int i = 0; i < 1 * 64 * 22 * 22; i++)
    {
        ASSERT_NEAR(output3[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(Convolution2dLayer, test4)
{
    // torch.nn.Conv2d(kernel_size = 7, stride = 3, padding = 2)
    // Output4 shape :  torch.Size([1, 64, 37, 37])
    Convolution2DLayer convolution2d("convolution2d");
    convolution2d.kernel_h_ = 7;
    convolution2d.kernel_w_ = 7;
    convolution2d.stride_h_ = 3;
    convolution2d.stride_w_ = 3;
    convolution2d.padding_h_ = 2;
    convolution2d.padding_w_ = 2;
    convolution2d.use_bias_ = true;
    convolution2d.dilation_h_ = 1;
    convolution2d.dilation_w_ = 1;
    convolution2d.groups_ = 1;
    convolution2d.in_channels_ = 64;
    convolution2d.out_channels_ = 64;

    // weight
    std::vector<uint32_t> weight_shape = {49};
    Tensor::TensorPtr weight = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight->copy_from(weight4, weight->size());
    // bias
    std::vector<uint32_t> bias_shape = {64};
    Tensor::TensorPtr bias = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias->copy_from(bias4, bias->size());

    convolution2d.bias_ = {bias};
    convolution2d.weights_ = {weight};

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 37, 37};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    convolution2d.prepare_layer({input_tensor}, {output_tensor});
    convolution2d.forward_cpu();

    for (int i = 0; i < 1 * 64 * 37 * 37; i++)
    {
        ASSERT_NEAR(output4[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    inferx::read_data_from_txt("test/test_convolution2d/test_data/input.txt", input, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/output1.txt", output1, 1 * 64 * 55 * 55);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/output2.txt", output2, 1 * 64 * 28 * 28);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/output3.txt", output3, 1 * 64 * 22 * 22);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/output4.txt", output4, 1 * 64 * 37 * 37);
    // weight
    inferx::read_data_from_txt("test/test_convolution2d/test_data/weight1.txt", weight1, 9);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/weight2.txt", weight2, 9);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/weight3.txt", weight3, 25);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/weight4.txt", weight4, 49);
    // bias
    inferx::read_data_from_txt("test/test_convolution2d/test_data/bias1.txt", bias1, 64);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/bias2.txt", bias2, 64);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/bias3.txt", bias3, 64);
    inferx::read_data_from_txt("test/test_convolution2d/test_data/bias4.txt", bias4, 64);
    LOG(INFO) << "Read data from file successfully";
    return RUN_ALL_TESTS();
}