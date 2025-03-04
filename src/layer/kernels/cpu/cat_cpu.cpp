/**
 * @file cat_cpu.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "layer/kernels/cat.h"

#include <cstdint>
#include <glog/logging.h>

namespace inferx
{
namespace layer
{

StatusCode CatLayer::forward_cpu()
{
    const uint32_t dim = dim_;
    const uint32_t num_inputs = inputs_.size();
    // check dimenson crorret
    const auto shapes1 = inputs_[0]->shapes();
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        CHECK(inputs_[i]->shapes().size() == shapes1.size()) << "Cat operate on wrong dims tensor.";
    }

    uint32_t catted_dim = 0;

    for (uint32_t j = 0; j < num_inputs; j++)
    {
        const auto now_shapes = inputs_[j]->shapes();
        for (uint32_t i = 0; i < shapes1.size(); i++)
        {
            if (i == dim)
            {
                catted_dim += now_shapes[i];
                continue;
            }
            CHECK(shapes1[i] == now_shapes[i])
                << "torch.cat error, you should make sure that dimesons are the same except the cat dimeson.";
        }
    }

    std::vector<uint32_t> output_shapes = shapes1;
    output_shapes[dim] = catted_dim;
    // outputs_[0] = std::make_shared<Tensor>(DataType::DataTypeFloat32, output_shapes);

    uint32_t offset = 1;
    for (uint32_t i = dim + 1; i < output_shapes.size(); i++)
    {
        offset *= output_shapes[i];
    }

    uint32_t all_size = 1;
    for (uint32_t i = 0; i < dim; i++)
    {
        all_size *= output_shapes[i];
    }

    uint32_t catted_size = 0;
    for (uint32_t i = 0; i < all_size; i++)
    {
        for (uint32_t j = 0; j < num_inputs; j++)
        {
            const auto curr_shape = inputs_[j]->shapes();
            const auto input = inputs_[j]->ptr<float>() + i * curr_shape[dim] * offset;
            const auto output = outputs_[0]->ptr<float>() + catted_size;
            memcpy(output, input, curr_shape[dim] * offset * sizeof(float));
            catted_size += curr_shape[dim] * offset * sizeof(float);
        }
    }

    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx