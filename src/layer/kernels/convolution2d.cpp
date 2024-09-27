/**
 * @file convolution2d.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "layer/kernels/convolution2d.h"
#include "core/common.h"
#include "layer/layer_factory.h"

#include <glog/logging.h>

namespace inferx
{

namespace layer
{

Convolution2DLayer::Convolution2DLayer(std::string layer_name)
    : Layer(layer_name)
{
}

Convolution2DLayer::~Convolution2DLayer() {}

StatusCode Convolution2DLayer::prepare_layer(
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

StatusCode Convolution2DLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("dilation") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter dilation is none, check your params.";
        return StatusCode::Failed;
    }
    dilation_h_ = params.at("dilation").ai[0];
    dilation_w_ = params.at("dilation").ai[1];

    if (params.find("in_channels") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter in_channels is none, check your params.";
        return StatusCode::Failed;
    }

    if (params.find("out_channels") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter out_channels is none, check your params.";
        return StatusCode::Failed;
    }

    in_channels_ = params.at("in_channels").i;
    out_channels_ = params.at("out_channels").i;

    if (params.find("padding") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter padding is none, check your params.";
        return StatusCode::Failed;
    }

    padding_h_ = params.at("padding").ai[0];
    padding_w_ = params.at("padding").ai[1];

    if (params.find("use_bias") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter use_bias is none, check your params.";
        return StatusCode::Failed;
    }

    use_bias_ = params.at("use_bias").b;

    if (params.find("stride") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter stride is none, check your params.";
        return StatusCode::Failed;
    }

    stride_h_ = params.at("stride").ai[0];
    stride_w_ = params.at("stride").ai[1];

    if (params.find("kernel_size") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter kernel_size is none, check your params.";
        return StatusCode::Failed;
    }

    kernel_h_ = params.at("kernel_size").ai[0];
    kernel_w_ = params.at("kernel_size").ai[1];

    if (params.find("groups") == params.end())
    {
        LOG(ERROR) << "Convolution2D operator parameter groups is none, check your params.";
        return StatusCode::Failed;
    }
    groups_ = params.at("groups").i;

    return StatusCode::Success;
}
StatusCode Convolution2DLayer::load_model(const std::map<std::string, pnnx::Attribute>& attributes)
{

    // bias : out_channels
    if (use_bias_)
    {
        if (attributes.find("bias") == attributes.end())
        {
            LOG(ERROR) << "Convolution2D operator attribute bias is none, check your model.";
            return StatusCode::Failed;
        }
        const auto& bias = attributes.at("bias").data;
        const auto& bias_shape = attributes.at("bias").shape;

        // set bias
        CHECK_EQ(out_channels_, bias_shape[0]) << "Convolution2D bias shape must be the same as out_channels.";
        this->bias_.resize(1);
        this->bias_[0] = std::make_shared<Tensor>(out_channels_);
        this->bias_[0]->copy_from(bias.data(), out_channels_);
    }

    // weight_shape : outchannels * in_channels * kernel_h * kernel_w
    if (attributes.find("weight") == attributes.end())
    {
        LOG(ERROR) << "Convolution2D operator attribute weight is none, check your model.";
        return StatusCode::Failed;
    }
    const auto& weights = attributes.at("weight").data;
    const auto& weights_shape = attributes.at("weight").shape;
    CHECK_EQ(weights_shape.size(), 4) << "Convolution2D weight shape must be 4 dims.";
    CHECK_EQ(out_channels_, weights_shape[0]) << "Convolution2D weight shape[0] must be equal to out_channels.";

    this->weights_.resize(1);
    this->weights_[0]
        = std::make_shared<Tensor>(weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3]);
    this->weights_[0]->copy_from(
        weights.data(), weights_shape[0] * weights_shape[1] * weights_shape[2] * weights_shape[3]);

    return StatusCode::Success;
}
Layer* createConvolution2DInstance(std::string layer_name)
{
    return new Convolution2DLayer(layer_name);
}

LayerRegisterWrapper Convolution2DLayer_Register(createConvolution2DInstance, "nn.Conv2d");

} // namespace layer
} // namespace inferx