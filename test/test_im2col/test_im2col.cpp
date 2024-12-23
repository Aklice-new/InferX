#include "core/tensor.h"
#include "layer/kernels/convolution2d.h"
#include "utils/utils.h"

#include <cstdint>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>

// weight shape :  torch.Size([4, 4, 3, 3])
// bias shape :  torch.Size([4])
// All outputs are matched
// Input shape :  torch.Size([1, 4, 5, 5])
// Col shape :  torch.Size([25, 36])
// Output shape :  torch.Size([1, 4, 5, 5])
float input[1 * 4 * 5 * 5];

float col1[25 * 36];

float output1[1 * 4 * 5 * 5];

float weight1[4 * 4 * 3 * 3];

float bias1[4];

uint32_t bias_size = 4;
uint32_t weight_size = 4 * 4 * 3 * 3;

using namespace inferx::layer;
TEST(Im2col, test1)
{
    Convolution2DLayer convolution2d("convolution2d");
    convolution2d.kernel_h_ = 3;
    convolution2d.kernel_w_ = 3;
    convolution2d.stride_h_ = 1;
    convolution2d.stride_w_ = 1;
    convolution2d.padding_h_ = 1;
    convolution2d.padding_w_ = 1;
    convolution2d.use_bias_ = true;
    convolution2d.dilation_h_ = 1;
    convolution2d.dilation_w_ = 1;
    convolution2d.groups_ = 1;
    convolution2d.in_channels_ = 4;
    convolution2d.out_channels_ = 4;

    // weight
    std::vector<uint32_t> weight_shape = {weight_size};
    Tensor::TensorPtr weight = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape);
    weight->copy_from(weight1, weight_size);

    // bias
    std::vector<uint32_t> bias_shape = {bias_size};
    Tensor::TensorPtr bias = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape);
    bias->copy_from(bias1, bias_size);

    convolution2d.bias_ = {bias};
    convolution2d.weights_ = {weight};

    std::vector<uint32_t> input_shape = {1, 4, 5, 5};
    Tensor::TensorPtr input_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, input_shape);
    input_tensor->copy_from(input, input_tensor->size());
    std::vector<uint32_t> output_shape = {1, 4, 5, 5};
    Tensor::TensorPtr output_tensor = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shape);
    output_tensor->apply_data();
    convolution2d.prepare_layer({input_tensor}, {output_tensor});
    convolution2d.forward_cpu();

    auto check_tensor = convolution2d.img2col_;

    for (int i = 0; i < 25 * 36; i++)
    {
        ASSERT_NEAR(col1[i], check_tensor->ptr<float>()[i], 1e-3)
            << "Mismatch at index " << i << " expected: " << col1[i] << " got: " << check_tensor->ptr<float>()[i];
    }

    for (int i = 0; i < 4 * 25; i++)
    {
        std::cout << &output_tensor->ptr<float>()[i] << ' ' << output_tensor->ptr<float>()[i] << ' ' << output1[i]
                  << ' ' << i / 25 << ' ' << i % 25 << std::endl;
        // ASSERT_NEAR(output1[i], output_tensor->ptr<float>()[i], 1e-3)
        //     << "Mismatch at index " << i << " expected: " << output1[i] << " got: " << check_tensor->ptr<float>()[i];
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    inferx::read_data_from_txt("test/test_im2col/test_data/input.txt", input, 1 * 4 * 5 * 5);
    inferx::read_data_from_txt("test/test_im2col/test_data/col1.txt", col1, 25 * 36);
    inferx::read_data_from_txt("test/test_im2col/test_data/output1.txt", output1, 1 * 4 * 5 * 5);
    inferx::read_data_from_txt("test/test_im2col/test_data/weight1.txt", weight1, 4 * 4 * 3 * 3);
    inferx::read_data_from_txt("test/test_im2col/test_data/bias1.txt", bias1, 4);
    return RUN_ALL_TESTS();
}
