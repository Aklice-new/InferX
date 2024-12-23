/**
 * @file clamp_cuda.cu
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "layer/kernels/clamp.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;

StatusCode ClampLayer::forward_gpu()
{

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx