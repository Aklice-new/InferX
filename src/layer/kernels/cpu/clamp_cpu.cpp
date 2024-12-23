/**
 * @file clamp.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "layer/kernels/clamp.h"
#include "utils/utils.h"

#include <cstdint>
#include <glog/logging.h>

namespace inferx
{
namespace layer
{

StatusCode ClampLayer::forward_cpu()
{
    const float min_val = min_val_;
    const float max_val = max_val_;
    const uint32_t size = inputs_[0]->size();
    auto input = inputs_[0];
    auto output = outputs_[0];
#pragma omp parallel for num_threads(MAX_THREADS)
    for (int t = 0; t < MAX_THREADS; t++)
    {
        int start_id = t * size / MAX_THREADS;
        for (uint32_t i = 0; i < (size / MAX_THREADS); i++)
        {
            int idx = start_id + i;
            output->ptr<float>()[idx] = std::max(min_val, std::min(max_val, input->ptr<float>()[idx]));
        }
    }
    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx