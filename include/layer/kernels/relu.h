#ifndef _RELU_H_
#define _RELU_H_

#include "core/common.h"
#include "layer/kernels/activation_type.h"
#include "layer/layer.h"
namespace inferx
{

namespace layer
{
class ReluLayer : public Layer
{
public:
    ReluLayer(std::string layer_name);

    // 检查tensor的位置，并转发计算操作到cpu or gpu
    StatusCode forward() override;
    StatusCode forward_gpu() override;
    StatusCode forward_cpu() override;
    StatusCode prepare_layer() override;

    // 设置加载本层的参数
    StatusCode load_model() override;

private:
    ActivationType activation_type_;
};

} // namespace layer
} // namespace inferx
#endif