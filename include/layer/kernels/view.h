/**
 * @file view.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-24
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _VIEW_H_
#define _VIEW_H_

#include "core/common.h"
#include "layer/layer.h"
#include <cstdint>

namespace inferx
{
namespace layer
{
using namespace inferx::core;
class ViewLayer : public Layer
{
public:
    explicit ViewLayer(std::string layer_name);
    virtual ~ViewLayer() = default;
    StatusCode forward_cpu() override;
#ifdef ENABLE_CUDA
    StatusCode forward_gpu() override;
#endif
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    // StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;
private:
    std::vector<uint32_t> shapes_;
};

} // namespace layer
} // namespace inferx

#endif // _VIEW_H_