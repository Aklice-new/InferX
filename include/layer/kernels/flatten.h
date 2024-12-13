/**
 * @file flatten.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef _H_FLATTEN_H_
#define _H_FLATTEN_H_
#include "core/common.h"
#include "layer/layer.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;
class FlattenLayer : public Layer
{
public:
    explicit FlattenLayer(std::string name);
    virtual ~FlattenLayer() = default;
    StatusCode forward_cpu() override;
#ifdef ENABLE_CUDA
    StatusCode forward_gpu() override;
#endif
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    // StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;

private:
    uint32_t start_dim_ = 0;
    uint32_t end_dim_ = 0;
};
} // namespace layer
} // namespace inferx

#endif