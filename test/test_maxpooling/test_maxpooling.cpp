#include "core/tensor.h"
#include "layer/kernels/maxpooling.h"
#include "utils/utils.h"

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>

// Input shape :  torch.Size([1, 64, 112, 112])
float input[1 * 64 * 112 * 112];

// Output1 shape :  torch.Size([1, 64, 55, 55])
// Output2 shape :  torch.Size([1, 64, 28, 28])
// Output3 shape :  torch.Size([1, 64, 22, 22])
// Output4 shape :  torch.Size([1, 64, 37, 37])
float output1[1 * 64 * 55 * 55];
float output2[1 * 64 * 28 * 28];
float output3[1 * 64 * 22 * 22];
float output4[1 * 64 * 37 * 37];

using namespace inferx::layer;

TEST(MaxPoolingLayer, test1)
{
    // torch.nn.MaxPool2d(kernel_size=3, stride=2)
    // Output1 shape :  torch.Size([1, 64, 55, 55])

    MaxPoolingLayer maxpooling("maxpooling");
    maxpooling.pooling_size_h_ = 3;
    maxpooling.pooling_size_w_ = 3;
    maxpooling.stride_h_ = 2;
    maxpooling.stride_w_ = 2;
    maxpooling.padding_h_ = 0;
    maxpooling.padding_w_ = 0;
    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 55, 55};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    maxpooling.prepare_layer({input_tensor}, {output_tensor});
    maxpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 55 * 55; i++)
    {
        ASSERT_NEAR(output1[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(MaxPoolingLayer, test2)
{
    // torch.nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
    // Output2 shape :  torch.Size([1, 64, 28, 28])

    MaxPoolingLayer maxpooling("maxpooling");
    maxpooling.pooling_size_h_ = 3;
    maxpooling.pooling_size_w_ = 3;
    maxpooling.stride_h_ = 4;
    maxpooling.stride_w_ = 4;
    maxpooling.padding_h_ = 1;
    maxpooling.padding_w_ = 1;
    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 28, 28};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    maxpooling.prepare_layer({input_tensor}, {output_tensor});
    maxpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 28 * 28; i++)
    {
        ASSERT_NEAR(output2[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(MaxPoolingLayer, test3)
{
    // torch.nn.MaxPool2d(kernel_size = 5, stride = 5, padding = 1)
    // Output3 shape :  torch.Size([1, 64, 22, 22])
    MaxPoolingLayer maxpooling("maxpooling");
    maxpooling.pooling_size_h_ = 5;
    maxpooling.pooling_size_w_ = 5;
    maxpooling.stride_h_ = 5;
    maxpooling.stride_w_ = 5;
    maxpooling.padding_h_ = 1;
    maxpooling.padding_w_ = 1;
    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 22, 22};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    maxpooling.prepare_layer({input_tensor}, {output_tensor});
    maxpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 22 * 22; i++)
    {
        ASSERT_NEAR(output3[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(MaxPoolingLayer, test4)
{
    // torch.nn.MaxPool2d(kernel_size = 7, stride = 3, padding = 2)
    // Output4 shape :  torch.Size([1, 64, 37, 37])
    MaxPoolingLayer maxpooling("maxpooling");
    maxpooling.pooling_size_h_ = 7;
    maxpooling.pooling_size_w_ = 7;
    maxpooling.stride_h_ = 3;
    maxpooling.stride_w_ = 3;
    maxpooling.padding_h_ = 2;
    maxpooling.padding_w_ = 2;
    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 37, 37};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    maxpooling.prepare_layer({input_tensor}, {output_tensor});
    maxpooling.forward_cpu();

    for (int i = 0; i < 1 * 64 * 37 * 37; i++)
    {
        ASSERT_NEAR(output4[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    inferx::read_data_from_txt("test/test_maxpooling/test_data/input.txt", input, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_maxpooling/test_data/output1.txt", output1, 1 * 64 * 55 * 55);
    inferx::read_data_from_txt("test/test_maxpooling/test_data/output2.txt", output2, 1 * 64 * 28 * 28);
    inferx::read_data_from_txt("test/test_maxpooling/test_data/output3.txt", output3, 1 * 64 * 22 * 22);
    inferx::read_data_from_txt("test/test_maxpooling/test_data/output4.txt", output4, 1 * 64 * 37 * 37);
    return RUN_ALL_TESTS();
}