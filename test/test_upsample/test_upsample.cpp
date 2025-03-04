#include "layer/kernels/upsample.h"
#include "core/tensor.h"
#include "utils/utils.h"

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>

// Input shape :  torch.Size([1, 64, 112, 112])
float input[1 * 64 * 112 * 112];

// Output1 shape :  torch.Size([1, 64, 224, 224])
// Output2 shape :  torch.Size([1, 64, 336, 336])
// Output3 shape :  torch.Size([1, 64, 448, 448])
// Output4 shape :  torch.Size([1, 64, 560, 560])
float output1[1 * 64 * 224 * 224];
float output2[1 * 64 * 336 * 336];
float output3[1 * 64 * 448 * 448];
float output4[1 * 64 * 560 * 560];

using namespace inferx::layer;

TEST(UpsampleLayer, test1)
{
    UpsampleLayer upsample("upsample");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 224, 224};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();

    upsample.mode_ = UpsampleLayer::Mode::Nearest;
    upsample.scale_factor_h_ = 2;
    upsample.scale_factor_w_ = 2;

    upsample.prepare_layer({input_tensor}, {output_tensor});
    upsample.forward_cpu();

    for (int i = 0; i < 1 * 64 * 224 * 224; i++)
    {
        ASSERT_NEAR(output1[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(UpsampleLayer, test2)
{
    UpsampleLayer upsample("upsample");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 336, 336};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);

    upsample.mode_ = UpsampleLayer::Mode::Nearest;
    upsample.scale_factor_h_ = 3;
    upsample.scale_factor_w_ = 3;

    output_tensor->apply_data();
    upsample.prepare_layer({input_tensor}, {output_tensor});
    upsample.forward_cpu();

    for (int i = 0; i < 1 * 64 * 336 * 336; i++)
    {
        ASSERT_NEAR(output2[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(UpsampleLayer, test3)
{
    UpsampleLayer upsample("upsample");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 448, 448};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);

    upsample.mode_ = UpsampleLayer::Mode::Nearest;
    upsample.scale_factor_h_ = 4;
    upsample.scale_factor_w_ = 4;

    output_tensor->apply_data();
    upsample.prepare_layer({input_tensor}, {output_tensor});
    upsample.forward_cpu();

    for (int i = 0; i < 1 * 64 * 448 * 448; i++)
    {
        ASSERT_NEAR(output3[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(UpsampleLayer, test4)
{
    UpsampleLayer upsample("upsample");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 560, 560};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);

    upsample.mode_ = UpsampleLayer::Mode::Nearest;
    upsample.scale_factor_h_ = 5;
    upsample.scale_factor_w_ = 5;

    output_tensor->apply_data();
    upsample.prepare_layer({input_tensor}, {output_tensor});
    upsample.forward_cpu();

    for (int i = 0; i < 1 * 64 * 560 * 560; i++)
    {
        ASSERT_NEAR(output4[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    inferx::read_data_from_txt("test/test_upsample/test_data/input.txt", input, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_upsample/test_data/output1.txt", output1, 1 * 64 * 224 * 224);
    inferx::read_data_from_txt("test/test_upsample/test_data/output2.txt", output2, 1 * 64 * 336 * 336);
    inferx::read_data_from_txt("test/test_upsample/test_data/output3.txt", output3, 1 * 64 * 448 * 448);
    inferx::read_data_from_txt("test/test_upsample/test_data/output4.txt", output4, 1 * 64 * 560 * 560);

    return RUN_ALL_TESTS();
}