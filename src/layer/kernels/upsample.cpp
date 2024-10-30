/**
 * @file upsample.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-24
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "core/common.h"
#include "layer/kernels/upsample.h"
#include "layer/layer_factory.h"

#include <cstdint>
#include <glog/logging.h>

namespace inferx
{
namespace layer
{
UpsampleLayer::UpsampleLayer(std::string name)
    : Layer(name)
{
}

StatusCode UpsampleLayer::prepare_layer(
    const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs)
{
    this->inputs_ = inputs;
    this->outputs_ = outputs;
    for (auto input : this->inputs_)
    {
        if (input->raw_ptr() == nullptr)
        {
            input->apply_data();
        }
    }
    for (auto output : this->outputs_)
    {
        if (output->raw_ptr() == nullptr)
        {
            output->apply_data();
        }
    }
    return StatusCode::Success;
}
StatusCode UpsampleLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("mode") == params.end())
    {
        LOG(ERROR) << "Upsample operator param shape is none, check your model.";
        return StatusCode::Failed;
    }
    auto mode = params.at("mode").s;
    if (mode == "nearest")
    {
        mode_ = Mode::Nearest;
    }
    else if (mode == "bilinear")
    {
        mode_ = Mode::Bilinear;
    }
    else
    {
        LOG(ERROR) << "Upsample operator mode is not supported, check your model.";
        return StatusCode::Failed;
    }
    if (params.find("scale_factor") == params.end())
    {
        LOG(ERROR) << "Upsample operator param shape is none, check your model.";
        return StatusCode::Failed;
    }
    auto scale_factor = params.at("scale_factor").ai;
    this->scale_factor_h_ = static_cast<uint32_t>(scale_factor[0]);
    this->scale_factor_w_ = static_cast<uint32_t>(scale_factor[1]);

    return StatusCode::Success;
}
Layer* createUpsampleInstance(std::string layer_name)
{
    return new UpsampleLayer(layer_name);
}

LayerRegisterWrapper UpsampleLayer_Register(createUpsampleInstance, "nn.Upsample");

} // namespace layer
} // namespace inferx