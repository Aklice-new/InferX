#ifndef _BATCHNORM_2D_H_
#define _BATCHNORM_2D_H_

#include "core/common.h"
#include "core/tensor.h"
#include "layer/layer.h"

namespace inferx
{
namespace layer
{

class BatchNorm2DLayer : public Layer
{
public:
    explicit BatchNorm2DLayer(std::string name);
    virtual ~BatchNorm2DLayer() = default;
    StatusCode forward_cpu() override;
    StatusCode forward_gpu() override;
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;

private:
    float eps_ = 1e-5;
    uint32_t num_features_ = 0;
    Tensor::TensorPtr mean_;
    Tensor::TensorPtr var_;
    std::vector<float> affine_gamma_;
    std::vector<float> affine_beta_;
};

} // namespace layer
} // namespace inferx

#endif