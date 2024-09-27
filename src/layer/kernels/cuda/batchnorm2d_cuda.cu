#include "core/common.h"
#include "layer/kernels/batchnorm2d.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;

StatusCode BatchNorm2DLayer::forward_gpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx