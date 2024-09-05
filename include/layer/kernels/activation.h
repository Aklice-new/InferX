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

    // 检查tensor的位置，并转发计算操作到cpu or gpu
    StatusCode forward_gpu() override;
    StatusCode forward_cpu() override;
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;

    // 设置加载本层的参数
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;

private:
    ActivationType activation_type_ = ActivationType::ActivationType_Unknow;
};

} // namespace layer
} // namespace inferx

#endif