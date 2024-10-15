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
#include "layer/kernels/flatten.h"
#include "layer/layer_factory.h"

#include <cstdint>
#include <glog/logging.h>
#include <sys/types.h>

namespace inferx
{
namespace layer
{

FlattenLayer::FlattenLayer(std::string name)
    : Layer(name)
{
}

StatusCode FlattenLayer::prepare_layer(
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

StatusCode FlattenLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{

    if (params.find("start_dim") == params.end())
    {
        LOG(ERROR) << "Flatten operator param start_dim is none, check your model.";
        return StatusCode::Failed;
    }
    int start_dim = params.at("start_dim").i;

    if (params.find("end_dim") == params.end())
    {
        LOG(ERROR) << "Flatten operator param end_dim is none, check your model.";
        return StatusCode::Failed;
    }

    int end_dim = params.at("end_dim").i;
    int total_dim = 4;
    if (start_dim < 0)
    {
        start_dim = total_dim + start_dim;
    }
    start_dim_ = static_cast<uint32_t>(start_dim);
    if (end_dim < 0)
    {
        end_dim = total_dim + end_dim;
    }
    end_dim_ = static_cast<uint32_t>(end_dim);

    // LOG(INFO) << "Flatten operator param start_dim: " << start_dim_ << " end_dim: " << end_dim_;
    return StatusCode::Success;
}

Layer* createFlattenInstance(std::string layer_name)
{
    return new FlattenLayer(layer_name);
}

LayerRegisterWrapper FlattenLayer_Register(createFlattenInstance, "torch.flatten");

} // namespace layer
} // namespace inferx