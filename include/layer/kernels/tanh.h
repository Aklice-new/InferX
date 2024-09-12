#ifndef _TANH_H_
#define _TANH_H_

#include "layer/kernels/activation.h"

namespace inferx
{
namespace layer
{
class TanHLayer : public ActivationLayer
{
public:
    TanHLayer(std::string layer_name);
};
} // namespace layer
} // namespace inferx
#endif