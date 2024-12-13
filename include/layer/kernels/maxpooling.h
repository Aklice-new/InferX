#ifndef _MAXPOOLING_H_
#define _MAXPOOLING_H_
#include "core/common.h"
#include "layer/layer.h"

namespace inferx
{
namespace layer
{
class MaxPoolingLayer : public Layer
{
public:
    explicit MaxPoolingLayer(std::string name);

    virtual ~MaxPoolingLayer() = default;
    StatusCode forward_cpu() override;
#ifdef ENABLE_CUDA
    StatusCode forward_gpu() override;
#endif
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;

    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    // StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;
    // private:
public:
    uint32_t padding_h_ = 0;
    uint32_t padding_w_ = 0;
    uint32_t pooling_size_h_ = 0;
    uint32_t pooling_size_w_ = 0;
    uint32_t stride_h_ = 1;
    uint32_t stride_w_ = 1;
};
} // namespace layer
} // namespace inferx

#endif