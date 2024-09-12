#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include "layer/kernels/activation.h"

namespace inferx
{
namespace layer
{
class SigmoidLayer : public ActivationLayer
{
public:
    SigmoidLayer(std::string layer_name);
};
} // namespace layer
} // namespace inferx
#endif