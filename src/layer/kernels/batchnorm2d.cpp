
/**
 * @file batchnorm2d.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "layer/kernels/batchnorm2d.h"
#include "core/common.h"
#include "layer/layer_factory.h"

#include <cstdint>
#include <glog/logging.h>
#include <sys/types.h>

namespace inferx
{
namespace layer
{

BatchNorm2DLayer::BatchNorm2DLayer(std::string name)
    : Layer(name)
{
}

StatusCode BatchNorm2DLayer::prepare_layer(
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

StatusCode BatchNorm2DLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("eps") == params.end())
    {
        LOG(ERROR) << "BatchNorm operator param eps is none, check your model.";
    }
    eps_ = params.at("eps").f;

    if (params.find("num_features") == params.end())
    {
        LOG(ERROR) << "BatchNorm operator param num_features is none, check your model.";
    }
    num_features_ = params.at("num_features").ai[0];

    return StatusCode::Success;
}

StatusCode BatchNorm2DLayer::load_model(const std::map<std::string, pnnx::Attribute>& attributes)
{

    if (attributes.find("running_mean") == attributes.end())
    {
        LOG(ERROR) << "BatchNorm operator attribute running_mean is none, check your model.";
        return StatusCode::Failed;
    }
    std::vector<uint32_t> num_features_shape = {num_features_};
    // load mean values
    mean_ = std::make_shared<Tensor>(DataType::DataTypeFloat32, num_features_shape);
    auto mean_ptr = reinterpret_cast<const float*>(attributes.at("running_mean").data.data());
    mean_->copy_from(reinterpret_cast<const void*>(mean_ptr), num_features_);
    if (attributes.find("running_var") == attributes.end())
    {
        LOG(ERROR) << "BatchNorm operator attribute running_var is none, check your model.";
        return StatusCode::Failed;
    }
    // load variance values
    var_ = std::make_shared<Tensor>(DataType::DataTypeFloat32, num_features_shape);
    auto var_ptr = reinterpret_cast<const float*>(attributes.at("running_var").data.data());
    var_->copy_from(reinterpret_cast<const void*>(var_ptr), num_features_);

    if (attributes.find("weight") == attributes.end())
    {
        LOG(ERROR) << "BatchNorm operator attribute weight is none, check your model.";
        return StatusCode::Failed;
    }
    // load gamma values
    affine_gamma_.resize(num_features_);
    auto gamma_ptr = reinterpret_cast<const float*>(attributes.at("weight").data.data());
    for (uint32_t i = 0; i < num_features_; i++)
    {
        float value = gamma_ptr[i];
        affine_gamma_[i] = value;
    }
    if (attributes.find("bias") == attributes.end())
    {
        LOG(ERROR) << "BatchNorm operator attribute bias is none, check your model.";
        return StatusCode::Failed;
    }
    // load beta values
    affine_beta_.resize(num_features_);
    auto beta_ptr = reinterpret_cast<const float*>(attributes.at("bias").data.data());
    for (uint32_t i = 0; i < num_features_; i++)
    {
        float value = beta_ptr[i];
        affine_beta_[i] = value;
    }
    return StatusCode::Success;
}

Layer* createBatchNorm2DInstance(std::string layer_name)
{
    return new BatchNorm2DLayer(layer_name);
}

LayerRegisterWrapper BatchNorm2DLayer_Register(createBatchNorm2DInstance, "nn.BatchNorm2d");

} // namespace layer
} // namespace inferx