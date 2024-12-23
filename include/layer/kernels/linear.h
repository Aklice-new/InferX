/**
 * @file linear.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _LINEAR_H_
#define _LINEAR_H_

#include "core/tensor.h"
#include "layer/layer.h"

#include <string>

namespace inferx
{
namespace layer
{

class LinearLayer : public Layer
{
public:
    explicit LinearLayer(std::string layer_name);
    ~LinearLayer() = default;
    StatusCode forward_cpu() override;
#ifdef ENABLE_CUDA
    StatusCode forward_gpu() override;
#endif
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;

    // private:
public:
    uint32_t in_features_ = 0, out_features_ = 0;
    bool use_bias_ = false;
    Tensor::TensorPtr weights_;
    Tensor::TensorPtr bias_;
};
} // namespace layer
} // namespace inferx

#endif