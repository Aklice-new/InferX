/**
 * @file upsample.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef _UPSMAPLE_H_
#define _UPSMAPLE_H_

#include "core/common.h"
#include "layer/layer.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;
class UpsampleLayer : public Layer
{
public:
    enum Mode
    {
        Nearest = 0,
        Bilinear = 1
    };
    explicit UpsampleLayer(std::string layer_name);
    virtual ~UpsampleLayer() = default;
    StatusCode forward_cpu() override;
    StatusCode forward_gpu() override;
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;
    StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes) override;

private:
    Mode mode_;
    uint32_t scale_factor_h_, scale_factor_w_;
};
} // namespace layer
} // namespace inferx

#endif //_UPSMAPLE_H_