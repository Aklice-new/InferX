#ifndef _HARD_SWISH_H_
#define _HARD_SWISH_H_

#include "layer/kernels/activation.h"

namespace inferx
{
namespace layer
{
class HardSwishLayer : public ActivationLayer
{
public:
    HardSwishLayer(std::string layer_name);
};

} // namespace layer
} // namespace inferx

#endif