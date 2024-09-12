#ifndef _RELU6_H_
#define _RELU6_H_

#include "layer/kernels/activation.h"

namespace inferx
{
namespace layer
{
class Relu6Layer : public ActivationLayer
{
public:
    Relu6Layer(std::string layer_name);
};
} // namespace layer
} // namespace inferx
#endif