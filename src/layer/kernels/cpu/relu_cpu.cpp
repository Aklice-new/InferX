#include "core/common.h"
#include "layer/kernels/relu.h"
namespace inferx
{
namespace layer
{

StatusCode ReluLayer::forward_cpu()
{
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx
