#ifndef _CONVOLUTION_2D_H_
#define _CONVOLUTION_2D_H_

#include "core/allocator.h"
#include "core/tensor.h"
#include "core/common.h"
#include "core/allocator.h"
#include "layer/layer.h"
#include <cstdint>

namespace inferx
{
namespace layer
{
using namespace inferx::core;

class Convolution2DLayer : public Layer
{
public:
    explicit Convolution2DLayer(std::string layer_name);
    virtual ~Convolution2DLayer();
    StatusCode forward_gpu() override;
    StatusCode forward_cpu() override;
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;

    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;

private:
    bool use_bias_ = false;

    uint32_t groups_;

    uint32_t in_channels_;
    uint32_t out_channels_;

    uint32_t padding_h_;
    uint32_t padding_w_;

    uint32_t stride_h_;
    uint32_t stride_w_;

    uint32_t dilation_h_;
    uint32_t dilation_w_;

    uint32_t kernel_h_;
    uint32_t kernel_w_;

    uint32_t output_h_;
    uint32_t output_w_;

    std::vector<Tensor::TensorPtr> weights_;
    std::vector<Tensor::TensorPtr> bias_;
};
} // namespace layer
} // namespace inferx

#endif // _CONVOLUTION_2D_H_