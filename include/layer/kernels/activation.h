#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "layer/layer.h"

namespace inferx
{
using namespace inferx::layer;
namespace activation
{
enum class ActivationType
{
    ActivationType_Relu6,
    ActivationType_Relu,
    ActivationType_Sigmoid,
    // ActivationType_LeakyRelu,
    // ActivationType_Prelu,
    // ActivationType_Elu,
    ActivationType_Silu,
    // ActivationType_Swish,
    // ActivationType_Hswish,
    // ActivationType_Hsigmoid,
    ActivationType_HardSigmoid,
    ActivationType_HardSwish,
    ActivationType_Unknow
};

class Activation : public Layer
{
};

} // namespace activation
} // namespace inferx

#endif