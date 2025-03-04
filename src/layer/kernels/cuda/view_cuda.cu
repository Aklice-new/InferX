#include "core/common.h"
#include "layer/kernels/view.h"
#include <cstdint>
#include <limits>

namespace inferx
{

namespace layer
{
using namespace core;
StatusCode ViewLayer::forward_gpu()
{
    return forward_cpu();
}
} // namespace layer
} // namespace inferx