/**
 * @file permute.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "core/common.h"
#include "layer/kernels/permute.h"
#include "layer/layer_factory.h"

#include <cstdint>
#include <glog/logging.h>

namespace inferx
{
namespace layer
{
PermuteLayer::PermuteLayer(std::string name)
    : Layer(name)
{
}

StatusCode PermuteLayer::prepare_layer(
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

StatusCode PermuteLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("order") == params.end())
    {
        LOG(ERROR) << "Permute operator param shape is none, check your model.";
        return StatusCode::Failed;
    }
    auto order = params.at("order").ai;
    for (auto i : order)
    {
        order_.push_back(i);
    }
    return StatusCode::Success;
}

Layer* createPermuteInstance(std::string layer_name)
{
    return new PermuteLayer(layer_name);
}

LayerRegisterWrapper PemuteLayer_Register(createPermuteInstance, "torch.permute");

} // namespace layer
} // namespace inferx