#include "core/common.h"
#include "core/tensor.h"
#include "layer/kernels/linear.h"
#include "utils/utils.h"

#include <cstdint>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace inferx::layer;

// Input shape :  torch.Size([1, 512])
float input[512];
// Output1 shape :  torch.Size([1, 1000])
// Output2 shape :  torch.Size([1, 512])
// Output3 shape :  torch.Size([1, 256])
// Output4 shape :  torch.Size([1, 128])
float output1[1000];
float output2[512];
float output3[256];
float output4[128];
// Weight1 shape :  torch.Size([1000, 512])
// Weight2 shape :  torch.Size([512, 512])
// Weight3 shape :  torch.Size([256, 512])
// Weight4 shape :  torch.Size([128, 512])
float weight1[1000 * 512];
float weight2[512 * 512];
float weight3[256 * 512];
float weight4[128 * 512];
// Bias1 shape :  torch.Size([1000])
// Bias2 shape :  torch.Size([512])
// Bias3 shape :  torch.Size([256])
// Bias4 shape :  torch.Size([128])
float bias1[1000];
float bias2[512];
float bias3[256];
float bias4[128];

TEST(LinearLayer, test1)
{
    LinearLayer linear("linear");
    linear.in_features_ = 1000;
    linear.out_features_ = 512;
    linear.use_bias_ = true;
    // weight
    std::vector<uint32_t> weight_shape = {1000, 512};
    auto weight_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight_tensor->copy_from(weight1, weight_tensor->size());

    // bias
    std::vector<uint32_t> bias_shape = {1000};
    auto bias_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias_tensor->copy_from(bias1, bias_tensor->size());

    linear.bias_ = bias_tensor;
    linear.weights_ = weight_tensor;

    // input
    std::vector<uint32_t> input_shape = {1, 512};
    auto input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, input_tensor->size());
    // output
    std::vector<uint32_t> output_shape = {1, 1000};
    auto output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    linear.prepare_layer({input_tensor}, {output_tensor});
    linear.forward_cpu();

    for (int i = 0; i < 1000; i++)
    {
        EXPECT_NEAR(output1[i], output_tensor->ptr<float>()[i], 1e-5)
            << " Missmatch at index " << i << " Expected : " << output1[i]
            << " Actual : " << output_tensor->ptr<float>()[i];
    }
}

TEST(LinearLayer, test2)
{
    LinearLayer linear("linear");
    linear.in_features_ = 512;
    linear.out_features_ = 512;
    linear.use_bias_ = true;
    // weight
    std::vector<uint32_t> weight_shape = {512, 512};
    auto weight_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight_tensor->copy_from(weight2, weight_tensor->size());

    // bias
    std::vector<uint32_t> bias_shape = {512};
    auto bias_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias_tensor->copy_from(bias2, bias_tensor->size());

    linear.bias_ = bias_tensor;
    linear.weights_ = weight_tensor;

    // input
    std::vector<uint32_t> input_shape = {1, 512};
    auto input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, input_tensor->size());
    // output
    std::vector<uint32_t> output_shape = {1, 512};
    auto output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    linear.prepare_layer({input_tensor}, {output_tensor});
    linear.forward_cpu();

    for (int i = 0; i < 512; i++)
    {
        EXPECT_NEAR(output2[i], output_tensor->ptr<float>()[i], 1e-5)
            << " Missmatch at index " << i << " Expected : " << output2[i]
            << " Actual : " << output_tensor->ptr<float>()[i];
    }
}

TEST(LinearLayer, test3)
{
    LinearLayer linear("linear");
    linear.in_features_ = 512;
    linear.out_features_ = 256;
    linear.use_bias_ = true;
    // weight
    std::vector<uint32_t> weight_shape = {256, 512};
    auto weight_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight_tensor->copy_from(weight3, weight_tensor->size());

    // bias
    std::vector<uint32_t> bias_shape = {256};
    auto bias_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias_tensor->copy_from(bias3, bias_tensor->size());

    linear.bias_ = bias_tensor;
    linear.weights_ = weight_tensor;

    // input
    std::vector<uint32_t> input_shape = {1, 512};
    auto input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, input_tensor->size());
    // output
    std::vector<uint32_t> output_shape = {1, 256};
    auto output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    linear.prepare_layer({input_tensor}, {output_tensor});
    linear.forward_cpu();

    for (int i = 0; i < 256; i++)
    {
        EXPECT_NEAR(output3[i], output_tensor->ptr<float>()[i], 1e-5)
            << " Missmatch at index " << i << " Expected : " << output3[i]
            << " Actual : " << output_tensor->ptr<float>()[i];
    }
}

TEST(LinearLayer, test4)
{
    LinearLayer linear("linear");
    linear.in_features_ = 512;
    linear.out_features_ = 128;
    linear.use_bias_ = true;
    // weight
    std::vector<uint32_t> weight_shape = {128, 512};
    auto weight_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight_tensor->copy_from(weight4, weight_tensor->size());

    // bias
    std::vector<uint32_t> bias_shape = {128};
    auto bias_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias_tensor->copy_from(bias4, bias_tensor->size());

    linear.bias_ = bias_tensor;
    linear.weights_ = weight_tensor;

    // input
    std::vector<uint32_t> input_shape = {1, 512};
    auto input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, input_tensor->size());
    // output
    std::vector<uint32_t> output_shape = {1, 128};
    auto output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    linear.prepare_layer({input_tensor}, {output_tensor});
    linear.forward_cpu();

    for (int i = 0; i < 128; i++)
    {
        EXPECT_NEAR(output4[i], output_tensor->ptr<float>()[i], 1e-5)
            << " Missmatch at index " << i << " Expected : " << output4[i]
            << " Actual : " << output_tensor->ptr<float>()[i];
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    inferx::read_data_from_txt("test/test_linear/test_data/input.txt", input, 512);
    inferx::read_data_from_txt("test/test_linear/test_data/output1.txt", output1, 1000);
    inferx::read_data_from_txt("test/test_linear/test_data/output2.txt", output2, 512);
    inferx::read_data_from_txt("test/test_linear/test_data/output3.txt", output3, 256);
    inferx::read_data_from_txt("test/test_linear/test_data/output4.txt", output4, 128);
    inferx::read_data_from_txt("test/test_linear/test_data/weight1.txt", weight1, 1000 * 512);
    inferx::read_data_from_txt("test/test_linear/test_data/weight2.txt", weight2, 512 * 512);
    inferx::read_data_from_txt("test/test_linear/test_data/weight3.txt", weight3, 256 * 512);
    inferx::read_data_from_txt("test/test_linear/test_data/weight4.txt", weight4, 128 * 512);
    inferx::read_data_from_txt("test/test_linear/test_data/bias1.txt", bias1, 1000);
    inferx::read_data_from_txt("test/test_linear/test_data/bias2.txt", bias2, 512);
    inferx::read_data_from_txt("test/test_linear/test_data/bias3.txt", bias3, 256);
    inferx::read_data_from_txt("test/test_linear/test_data/bias4.txt", bias4, 128);
    return RUN_ALL_TESTS();
}