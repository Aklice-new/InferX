#include "core/common.h"
#include "layer/kernels/convolution2d.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;

StatusCode Convolution2DLayer::forward_gpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto weight = weights_[0];
    auto bias = bias_[0];

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx