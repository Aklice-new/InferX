#include "core/common.h"
#include "layer/kernels/upsample.h"
#include <cstdint>
#include <limits>

namespace inferx
{

namespace layer
{
using namespace core;
StatusCode UpsampleLayer::forward_gpu()
{
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx