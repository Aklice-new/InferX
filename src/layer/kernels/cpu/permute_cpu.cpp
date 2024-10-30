/**
 * @file permute_cpu.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "core/common.h"
#include "layer/kernels/permute.h"

#include <cstdint>
#include <glog/logging.h>
#include <sys/types.h>
#include <vector>

namespace inferx
{
namespace layer
{

uint32_t calSourceIndex(uint32_t now_idx, const std::vector<uint32_t>& new_strides,
    const std::vector<uint32_t>& old_strides, const std::vector<uint32_t>& order)
{
    // OffsetToNdIndex
    const uint32_t ndim = new_strides.size();
    std::vector<uint32_t> index(ndim, 0);
    auto remaining = now_idx;

    for (int i = 0; i < ndim; i++)
    {
        index[i] = remaining / new_strides[i];
        remaining %= new_strides[i];
    }
    // swap to old index
    std::vector<uint32_t> old_index(ndim, 0);
    for (int i = 0; i < ndim; i++)
    {
        old_index[i] = index[order[i]];
    }
    // NdindexToOffset
    uint32_t old_offset = 0;
    for (int i = 0; i < ndim; i++)
    {
        old_offset += old_index[i] * old_strides[i];
    }
    return old_offset;
}

StatusCode PermuteLayer::forward_cpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto in_shapes = input->shapes();
    auto in_strides = input->strides();
    auto out_shapes = output->shapes();
    auto out_strides = output->strides();
    const auto& order = this->order_;

    auto element_size = input->size();
    auto input_ptr = input->ptr<float>();
#pragma omp parallel for
    for (auto i = 0; i < element_size; i++)
    {
        auto old_offset = calSourceIndex(i, out_strides, in_strides, order);
        output->ptr<float>()[i] = input_ptr[old_offset];
    }

    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx