#include "core/common.h"
#include "layer/kernels/permute.h"
#include <cstdint>
#include <limits>

namespace inferx
{

namespace layer
{
using namespace core;
StatusCode PermuteLayer::forward_gpu()
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