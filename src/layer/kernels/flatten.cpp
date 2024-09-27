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

#include <cstddef>
#include <glog/logging.h>

namespace inferx
{
namespace layer
{

FlattenLayer::FlattenLayer(std::string name)
    : Layer(name)
{
}

StatusCode FlattenLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{

    if (params.find("start_dim") == params.end())
    {
        LOG(ERROR) << "Flatten operator param start_dim is none, check your model.";
        return StatusCode::Failed;
    }
    start_dim_ = params.at("start_dim").ai[0];

    if (params.find("end_dim") == params.end())
    {
        LOG(ERROR) << "Flatten operator param end_dim is none, check your model.";
        return StatusCode::Failed;
    }
    end_dim_ = params.at("end_dim").ai[0];

    return StatusCode::Success;
}

Layer* createFlattenInstance(std::string layer_name)
{
    return new FlattenLayer(layer_name);
}

LayerRegisterWrapper FlattenLayer_Register(createFlattenInstance, "torch.flatten");

} // namespace layer
} // namespace inferx