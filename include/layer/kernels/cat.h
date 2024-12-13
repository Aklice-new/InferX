/**
 * @file cat.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-09
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _CAT_H_
#define _CAT_H_

#include "core/common.h"
#include "layer/layer.h"

#include <string>

namespace inferx
{
namespace layer
{
class CatLayer : public Layer
{
public:
    explicit CatLayer(const std::string& name);
    virtual ~CatLayer() = default;
    StatusCode forward_cpu() override;
#ifdef ENABLE_CUDA
    StatusCode forward_gpu() override;
#endif
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;

private:
    uint32_t dim_ = 0;
};
} // namespace layer

} // namespace inferx

#endif