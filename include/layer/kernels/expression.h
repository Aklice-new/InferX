/**
 * @file expression.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-27
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _EXPRESSION_H_
#define _EXPRESSION_H_

#include "core/common.h"
#include "layer/layer.h"
#include "parser/parser.h"

namespace inferx
{
namespace layer
{
using namespace inferx::parser;
/**
 * @brief 现在支持的运算如下：
    tensor + tensor
    tensor * tensor
    tensor / num
    sqrt(tensor)
    当然后续还可以继续支持其他
 *
 */
class ExpressionLayer : public Layer
{
public:
    enum BinaryOP
    {
        ADD_OP,
        SUB_OP,
        MUL_OP,
        DIV_OP
    };
    explicit ExpressionLayer(const std::string& name);
    virtual ~ExpressionLayer() = default;
    StatusCode forward_cpu() override;
    StatusCode forward_gpu() override;
    StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs) override;
    StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params) override;

private:
    std::string expression_;
    std::unique_ptr<ExpressionParser> parser_;
    std::vector<std::shared_ptr<TokenNode>> inverse_polish_notation_;
};
} // namespace layer
} // namespace inferx

#endif