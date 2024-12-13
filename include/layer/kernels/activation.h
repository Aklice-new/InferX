#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_
#include "core/common.h"
#include "core/tensor.h"
#include "layer/layer.h"
namespace inferx
{
namespace layer
{

enum class ActivationType
{
    ActivationType_Relu6,
    ActivationType_Relu,
    ActivationType_Sigmoid,
    ActivationType_Silu,
    ActivationType_HardSigmoid,
    ActivationType_HardSwish,
    ActivationType_TanH,
    ActivationType_Unknow
};

class ActivationLayer : public Layer
{
public:
    ActivationLayer(std::string layer_name);
    virtual ~ActivationLayer() = default;
    // 检查tensor的位置，并转发计算操作到cpu or gpu
#ifdef ENABLE_CUDA
    StatusCode forward_gpu() override;
#endif
    StatusCode forward_cpu() override;
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;

protected:
    ActivationType activation_type_ = ActivationType::ActivationType_Unknow;
};

} // namespace layer
} // namespace inferx

#endif