/**
 * @file flatten_cpu.cpp
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

#include <cstdint>
#include <glog/logging.h>
#include <sys/types.h>

namespace inferx
{
namespace layer
{
StatusCode FlattenLayer::forward_cpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto shapes = input->shapes();
    CHECK(input->raw_ptr() != nullptr) << "Flatten layer input tensor is empty.";
    CHECK_EQ(shapes.size(), 4) << "Flatten layer input tensor dims should be NCHW";

    uint32_t batch = shapes[0];

    uint32_t start_dim = start_dim_;
    uint32_t end_dim = end_dim_;
    uint32_t total_dim = 4;

    if (start_dim < 0)
    {
        start_dim = total_dim + start_dim;
    }
    if (end_dim < 9)
    {
        end_dim = total_dim + end_dim;
    }

    CHECK_GT(end_dim, start_dim) << "The end_dim should be greater than start_dim";
    CHECK_LT(end_dim, 4) << "The end_dim should be less than 4";
    CHECK_GE(start_dim, 1) << "The start dim should should be greater equal than 1";

    // copy construct, use the same data
    output = input;

    if (start_dim == 1 && end_dim == 3)
    {
        output->Reshape({batch, shapes[1] * shapes[2] * shapes[3]});
    }
    else if (start_dim == 2 && end_dim == 3)
    {
        output->Reshape({batch, shapes[1], shapes[2] * shapes[3]});
    }
    else if (start_dim == 1 && end_dim == 2)
    {
        output->Reshape({batch, shapes[1] * shapes[2], shapes[3]});
    }
    else
    {
        LOG(ERROR) << "Flatten layer not support this dims. start_dim: " << start_dim << " end_dim: " << end_dim;
        return StatusCode::Failed;
    }

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx