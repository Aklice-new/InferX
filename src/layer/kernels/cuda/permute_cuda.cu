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
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx