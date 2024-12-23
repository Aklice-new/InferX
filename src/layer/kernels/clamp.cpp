/**
 * @file flatten.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "layer/kernels/clamp.h"
#include "layer/layer_factory.h"

#include <cstdint>
#include <glog/logging.h>
#include <sys/types.h>

namespace inferx
{
namespace layer
{

ClampLayer::ClampLayer(const std::string& name)
    : Layer(name)
{
}

StatusCode ClampLayer::prepare_layer(
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

StatusCode ClampLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{

    if (params.find("max") == params.end())
    {
        LOG(ERROR) << "Clamp operator param max is none, check your model.";
        return StatusCode::Failed;
    }
    float mx = params.at("max").f;

    if (params.find("min") == params.end())
    {
        LOG(ERROR) << "Clamp  operator param min is none, check your model.";
        return StatusCode::Failed;
    }
    float mn = params.at("min").f;

    min_val_ = mn;
    max_val_ = mx;

    // LOG(INFO) << "Flatten operator param start_dim: " << start_dim_ << " end_dim: " << end_dim_;
    return StatusCode::Success;
}

Layer* createClampInstance(std::string layer_name)
{
    return new ClampLayer(layer_name);
}

LayerRegisterWrapper ClampLayer_Register(createClampInstance, "torch.clamp");

} // namespace layer
} // namespace inferx