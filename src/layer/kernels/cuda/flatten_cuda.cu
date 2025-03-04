#include "core/common.h"
#include "layer/kernels/flatten.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;

StatusCode FlattenLayer::forward_gpu()
{
    // auto input = inputs_[0];
    // auto output = outputs_[0];

    return forward_cpu();
}
} // namespace layer
} // namespace inferx