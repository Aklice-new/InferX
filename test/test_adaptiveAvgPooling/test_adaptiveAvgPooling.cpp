#include "core/tensor.h"
#include "layer/kernels/adaptive_avgpooling.h"
#include "utils/utils.h"

#include <cstdint>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>

// Input shape :  torch.Size([1, 64, 112, 112])
float input[1 * 64 * 112 * 112];

// Output1 shape :  torch.Size([1, 64, 54, 54])
// Output2 shape :  torch.Size([1, 64, 37, 37])
// Output3 shape :  torch.Size([1, 64, 1, 1])
// Output4 shape :  torch.Size([1, 64, 99, 99])
float output1[1 * 64 * 54 * 54];
float output2[1 * 64 * 37 * 37];
float output3[1 * 64 * 1 * 1];
float output4[1 * 64 * 99 * 99];

using namespace inferx::layer;

TEST(AdaptiveAvgPooling, test1)
{
    AdaptiveAvgPoolingLayer adaptive_avgpooling("adaptive_avgpooling");
    adaptive_avgpooling.output_height_ = 54;
    adaptive_avgpooling.output_width_ = 54;

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 54, 54};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    adaptive_avgpooling.prepare_layer({input_tensor}, {output_tensor});
    adaptive_avgpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 54 * 54; i++)
    {
        ASSERT_NEAR(output1[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}
TEST(AdaptiveAvgPooling, test2)
{
    AdaptiveAvgPoolingLayer adaptive_avgpooling("adaptive_avgpooling");
    adaptive_avgpooling.output_height_ = 37;
    adaptive_avgpooling.output_width_ = 37;

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 37, 37};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    adaptive_avgpooling.prepare_layer({input_tensor}, {output_tensor});
    adaptive_avgpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 37 * 37; i++)
    {
        ASSERT_NEAR(output2[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}
TEST(AdaptiveAvgPooling, test3)
{
    AdaptiveAvgPoolingLayer adaptive_avgpooling("adaptive_avgpooling");
    adaptive_avgpooling.output_height_ = 1;
    adaptive_avgpooling.output_width_ = 1;

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 1, 1};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    adaptive_avgpooling.prepare_layer({input_tensor}, {output_tensor});
    adaptive_avgpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 1 * 1; i++)
    {
        ASSERT_NEAR(output3[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}
TEST(AdaptiveAvgPooling, test4)
{
    AdaptiveAvgPoolingLayer adaptive_avgpooling("adaptive_avgpooling");
    adaptive_avgpooling.output_height_ = 99;
    adaptive_avgpooling.output_width_ = 99;

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 99, 99};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    adaptive_avgpooling.prepare_layer({input_tensor}, {output_tensor});
    adaptive_avgpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 99 * 99; i++)
    {
        ASSERT_NEAR(output4[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    inferx::read_data_from_txt("test/test_adaptiveAvgPooling/test_data/input.txt", input, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_adaptiveAvgPooling/test_data/output1.txt", output1, 1 * 64 * 54 * 54);
    inferx::read_data_from_txt("test/test_adaptiveAvgPooling/test_data/output2.txt", output2, 1 * 64 * 37 * 37);
    inferx::read_data_from_txt("test/test_adaptiveAvgPooling/test_data/output3.txt", output3, 1 * 64 * 1 * 1);
    inferx::read_data_from_txt("test/test_adaptiveAvgPooling/test_data/output4.txt", output4, 1 * 64 * 99 * 99);

    return RUN_ALL_TESTS();
}