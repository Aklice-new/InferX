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

class ExpressionLayer : public Layer
{
public:
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