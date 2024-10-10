/**
 * @file cat.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "layer/kernels/cat.h"
#include "core/common.h"
#include "layer/layer_factory.h"

#include <glog/logging.h>

namespace inferx
{
namespace layer
{

CatLayer::CatLayer(const std::string& name)
    : Layer(name)
{
}

StatusCode CatLayer::prepare_layer(
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

StatusCode CatLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("dim") == params.end())
    {
        LOG(ERROR) << "Cat operator parameter dim is none, check your params.";
        return StatusCode::Failed;
    }
    dim_ = params.at("dim").ai[0];
    const int input_dims = inputs_[0]->shapes().size();
    dim_ = dim_ < 0 ? dim_ + input_dims : dim_;
    CHECK(dim_ >= 0 && dim_ < input_dims) << "Cat operator dim is out of range.";
    return StatusCode::Success;
}

Layer* createCatLayerInstance(std::string layer_name)
{
    return new CatLayer(layer_name);
}

LayerRegisterWrapper CatLayer_Register(createCatLayerInstance, "torch.cat");

} // namespace layer
} // namespace inferx