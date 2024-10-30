/**
 * @file view_cpu.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "layer/kernels/view.h"

#include <cstdint>
#include <glog/logging.h>
#include <sys/types.h>

namespace inferx
{
namespace layer
{
StatusCode ViewLayer::forward_cpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto shapes = input->shapes();
    CHECK(input->raw_ptr() != nullptr) << "View layer input tensor is empty.";
    // CHECK_EQ(shapes.size(), 4) << "View layer input tensor dims should";
    uint32_t new_size = 1;
    for (auto s : shapes_)
    {
        new_size *= s;
    }
    CHECK(new_size == input->size()) << "View layer input tensor size is not equal to output tensor size.";
    // output->apply_data();
    output->copy_from(input->raw_ptr(), input->size());
    output->Reshape(this->shapes_);

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx