/**
 * @file permute.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef _PERMUTE_H_
#define _PERMUTE_H_

#include "core/common.h"
#include "layer/layer.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;
class PermuteLayer : public Layer
{
public:
    PermuteLayer(std::string name);
    ~PermuteLayer() = default;
    StatusCode forward_cpu() override;
#ifdef ENABLE_CUDA
    StatusCode forward_gpu() override;
#endif
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    // StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;
private:
    std::vector<uint32_t> order_;
};
} // namespace layer
} // namespace inferx

#endif // _PERMUTE_H_