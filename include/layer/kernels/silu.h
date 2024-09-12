#ifndef _SILU_H_
#define _SILU_H_

#include "layer/kernels/activation.h"

namespace inferx
{
namespace layer
{
class SiluLayer : public ActivationLayer
{
public:
    SiluLayer(std::string layer_name);
};
} // namespace layer
} // namespace inferx
#endif