#include "core/tensor.h"
#include "layer/kernels/hard_swish.h"
#include "utils/utils.h"

#include <cstdint>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>

// Input shape :  torch.Size([1, 64, 112, 112])
float input[1 * 64 * 112 * 112];

// Output1 shape :  torch.Size([1, 64, 112, 112])
// Output2 shape :  torch.Size([1, 64, 112, 112])
// Output3 shape :  torch.Size([1, 64, 112, 112])
// Output4 shape :  torch.Size([1, 64, 112, 112])
float output1[1 * 64 * 112 * 112];
float output2[1 * 64 * 112 * 112];
float output3[1 * 64 * 112 * 112];
float output4[1 * 64 * 112 * 112];

using namespace inferx::layer;

TEST(HardSwishLayer, test1)
{
    HardSwishLayer sigmoid("sigmoid");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 112, 112};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    sigmoid.prepare_layer({input_tensor}, {output_tensor});
    sigmoid.forward_cpu();

    for (int i = 0; i < 1 * 64 * 112 * 112; i++)
    {
        ASSERT_NEAR(output1[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(HardSwishLayer, test2)
{
    HardSwishLayer sigmoid("sigmoid");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 112, 112};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    sigmoid.prepare_layer({input_tensor}, {output_tensor});
    sigmoid.forward_cpu();

    for (int i = 0; i < 1 * 64 * 112 * 112; i++)
    {
        ASSERT_NEAR(output2[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(HardSwishLayer, test3)
{
    HardSwishLayer sigmoid("sigmoid");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 112, 112};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    sigmoid.prepare_layer({input_tensor}, {output_tensor});
    sigmoid.forward_cpu();

    for (int i = 0; i < 1 * 64 * 112 * 112; i++)
    {
        ASSERT_NEAR(output3[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

TEST(HardSwishLayer, test4)
{
    HardSwishLayer sigmoid("sigmoid");

    std::vector<uint32_t> input_shape = {1, 64, 112, 112};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, 1 * 64 * 112 * 112);
    std::vector<uint32_t> output_shape = {1, 64, 112, 112};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    sigmoid.prepare_layer({input_tensor}, {output_tensor});
    sigmoid.forward_cpu();

    for (int i = 0; i < 1 * 64 * 112 * 112; i++)
    {
        ASSERT_NEAR(output4[i], output_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << output_tensor->ptr<float>()[i];
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    inferx::read_data_from_txt("test/test_hardswish/test_data/input.txt", input, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_hardswish/test_data/output1.txt", output1, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_hardswish/test_data/output2.txt", output2, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_hardswish/test_data/output3.txt", output3, 1 * 64 * 112 * 112);
    inferx::read_data_from_txt("test/test_hardswish/test_data/output4.txt", output4, 1 * 64 * 112 * 112);

    return RUN_ALL_TESTS();
}