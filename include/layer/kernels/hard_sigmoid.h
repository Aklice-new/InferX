#ifndef _HARD_SIGMOID_H_
#define _HARD_SIGMOID_H_

#include "layer/kernels/activation.h"

namespace inferx
{
namespace layer
{
class HardSigmoidLayer : public ActivationLayer
{
public:
    HardSigmoidLayer(std::string layer_name);
};

} // namespace layer
} // namespace inferx

#endif