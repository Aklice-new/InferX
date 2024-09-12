#ifndef _RELU_H_
#define _RELU_H_

#include "layer/kernels/activation.h"

namespace inferx
{
namespace layer
{
class ReluLayer : public ActivationLayer
{
public:
    ReluLayer(std::string layer_name);
};
} // namespace layer
} // namespace inferx
#endif