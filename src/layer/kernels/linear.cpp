/**
 * @file linear.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "layer/kernels/linear.h"
#include "core/common.h"
#include "core/tensor.h"
#include "layer/layer_factory.h"

#include <cstdint>
#include <glog/logging.h>
#include <memory>

namespace inferx
{
namespace layer
{

LinearLayer::LinearLayer(std::string layer_name)
    : Layer(layer_name)
{
}

StatusCode LinearLayer::prepare_layer(
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

StatusCode LinearLayer::load_model(const std::map<std::string, pnnx::Attribute>& attributes)
{
    if (attributes.empty())
    {
        LOG(ERROR) << "LinearLayer: No attributes found, check your model file";
        return StatusCode::Failed;
    }

    if (use_bias_)
    {
        if (attributes.find("bias") == attributes.end())
        {
            LOG(ERROR) << "LinearLayer: No bias found, check your model file";
            return StatusCode::Failed;
        }
        const auto& bias = attributes.at("bias").data;
        const auto& bias_shape = attributes.at("bias").shape;
        std::vector<uint32_t> bias_shape_32 = {1, out_features_};
        bias_ = std::make_shared<Tensor>(DataType::DataTypeFloat32, bias_shape_32);
        CHECK_EQ(bias_shape[0], out_features_) << " bias shape should be the same as in_features";
        // LOG(INFO) << "bias data size " << bias.size();
        // LOG(INFO) << "bias shape " << bias_shape[0] << " " << bias_shape[1];
        // LOG(INFO) << "out_features_ " << out_features_;
        // LOG(INFO) << "in_features_ " << in_features_;
        bias_->copy_from(reinterpret_cast<const void*>(bias.data()), out_features_);
    }

    if (attributes.find("weight") == attributes.end())
    {
        LOG(ERROR) << "LinearLyaer : No weight found, check your modle file";
        return StatusCode::Failed;
    }

    const auto& weight = attributes.at("weight").data;
    const auto& weight_shape = attributes.at("weight").shape;
    CHECK_EQ(in_features_, weight_shape[1]) << "LinearLayer : the weight width should be the same as in_features";
    CHECK_EQ(out_features_, weight_shape[0]) << "LinearLyaer : the weight height should be the same as out_features";

    std::vector<uint32_t> weight_shape_32 = {out_features_, in_features_};
    weights_ = std::make_shared<Tensor>(DataType::DataTypeFloat32, weight_shape_32);
    weights_->copy_from(reinterpret_cast<const void*>(weight.data()), in_features_ * out_features_);
    return StatusCode::Success;
}

StatusCode LinearLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.empty())
    {
        LOG(ERROR) << "LinearLayer: No parameters found, check your model file";
        return StatusCode::Failed;
    }

    if (params.find("bias") == params.end())
    {
        LOG(ERROR) << "LinearLayer: No bias found, check your model file";
        return StatusCode::Failed;
    }
    use_bias_ = params.at("bias").b;

    if (params.find("in_features") == params.end())
    {
        LOG(ERROR) << "LinearLayer: No in_features found, check your model file";
        return StatusCode::Failed;
    }
    in_features_ = params.at("in_features").i;

    if (params.find("out_features") == params.end())
    {
        LOG(ERROR) << "LinearLayer: No out_features found, check your model file";
        return StatusCode::Failed;
    }
    out_features_ = params.at("out_features").i;

    return StatusCode::Success;
}

Layer* createLinearInstance(std::string layer_name)
{
    return new LinearLayer(layer_name);
}

LayerRegisterWrapper LinearLayer_Register(createLinearInstance, "nn.Linear");

} // namespace layer
} // namespace inferx