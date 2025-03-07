/**
 * @file expression.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "layer/kernels/expression.h"
#include "core/common.h"
#include "layer/layer_factory.h"

#include <glog/logging.h>

namespace inferx
{
namespace layer
{
ExpressionLayer::ExpressionLayer(const std::string& name)
    : Layer(name)
{
}

StatusCode ExpressionLayer::prepare_layer(
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

StatusCode ExpressionLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("expr") == params.end())
    {
        LOG(ERROR) << "Expression operator parameter expr is none, check your params.";
        return StatusCode::Failed;
    }
    expression_ = params.at("expr").s;
    parser_ = std::make_unique<ExpressionParser>(expression_);
    inverse_polish_notation_ = parser_->GenerateSyntaxTree();
    return StatusCode::Success;
}

Layer* createExpressionLayerInstance(std::string layer_name)
{
    return new ExpressionLayer(layer_name);
}

LayerRegisterWrapper ExpressionLayer_Register(createExpressionLayerInstance, "pnnx.Expression");

} // namespace layer
} // namespace inferx
