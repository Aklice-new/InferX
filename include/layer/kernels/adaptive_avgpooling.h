#ifndef _ADAPTIVE_AVGPOOLING_H_
#define _ADAPTIVE_AVGPOOLING_H_

#include "core/common.h"
#include "layer/layer.h"

namespace inferx
{
namespace layer
{

class AdaptiveAvgPoolingLayer : public Layer
{
public:
    explicit AdaptiveAvgPoolingLayer(std::string layer_name);
    virtual ~AdaptiveAvgPoolingLayer() = default;
    StatusCode forward_gpu() override;
    StatusCode forward_cpu() override;
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;

    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    // StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;

private:
    uint32_t output_height_ = 0;
    uint32_t output_width_ = 0;
};
} // namespace layer
} // namespace inferx

#endif