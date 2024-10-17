#include "core/common.h"
#include "core/tensor.h"
#include "layer/kernels/expression.h"

#include <cmath>
#include <stack>
#include <type_traits>
#include <variant>

#include <glog/logging.h>

namespace inferx
{
namespace layer
{

template <typename DType>
void tensor_element_add(DType* input1, DType* input2, DType* output, size_t size)
{
#pragma omp parallel for num_threads(size)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input1[i] + input2[i];
    }
}

template <typename DType>
void tensor_element_mul(DType* input1, DType* input2, DType* output, size_t size)
{
#pragma omp parallel for num_threads(size)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input1[i] * input2[i];
    }
}
template <typename DType>
void tensor_element_div(DType* input1, int divisor, DType* output, size_t size)
{
#pragma omp parallel for num_threads(size)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input1[i] / divisor;
    }
}
template <typename DType>
void tensor_element_sqrt(DType* input1, DType* output, size_t size)
{
#pragma omp parallel for num_threads(size)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = std::sqrt(input1[i]);
    }
}

std::vector<uint32_t> broadcast_shapes(std::vector<uint32_t>& shapes1, std::vector<uint32_t>& shapes2)
{
    // TODO: add some broadcast rules checking
    std::vector<uint32_t> broadcast_shape;
    auto size1 = shapes1.size();
    auto size2 = shapes2.size();
    auto max_size = std::max(size1, size2);
    for (size_t i = 0; i < max_size; i++)
    {
        auto shape1 = i < size1 ? shapes1[i] : 1;
        auto shape2 = i < size2 ? shapes2[i] : 1;
        auto shape = std::max(shape1, shape2);
        broadcast_shape.push_back(shape);
    }
    return broadcast_shape;
}

/**
 * @brief 根据逆波兰式计算表达式
 *
 */
StatusCode ExpressionLayer::forward_cpu()
{

    CHECK(parser_ != nullptr) << "Parser is null.";
    const auto inverse_polish_notation = inverse_polish_notation_; // parser_->GenerateSyntaxTree();
    CHECK(!inverse_polish_notation.empty()) << "Inverse polish notation is empty, check your expression.";
    using VariantType = std::variant<Tensor::TensorPtr, int>;
    std::stack<VariantType> op_stack;
    for (auto node : inverse_polish_notation)
    {
        if (node->token_.type == TokenType::TokenTensor)
        {
            auto tensor_idx = std::stoi(node->token_.value);
            auto tensor = inputs_[tensor_idx];
            op_stack.push(tensor);
        }
        else if (node->token_.type == TokenType::TokenNumber)
        {
            int number = std::stoi(node->token_.value);
            op_stack.push(number);
        }
        else if (node->token_.type == TokenType::TokenAdd || node->token_.type == TokenType::TokenMul
            || node->token_.type == TokenType::TokenDiv)
        {
            CHECK(op_stack.size() >= 2) << "Operator stack size is less than 2, but this operator needs 2 operband.";
            auto right = op_stack.top();
            op_stack.pop();
            auto left = op_stack.top();
            op_stack.pop();
            // using R_T = std::decay_t<decltype(right)>;
            // using L_T = std::decay_t<decltype(left)>;

            Tensor::TensorPtr output = nullptr;

            if (node->token_.type == TokenType::TokenAdd)
            {
                if (std::holds_alternative<Tensor::TensorPtr>(left) && std::holds_alternative<Tensor::TensorPtr>(right))
                {
                    auto tensor1 = std::get<Tensor::TensorPtr>(left);
                    auto tensor2 = std::get<Tensor::TensorPtr>(right);
                    auto shape1 = tensor1->shapes();
                    auto shape2 = tensor2->shapes();
                    if (shape1 == shape2)
                    {
                        output = std::make_shared<Tensor>(DataType::DataTypeFloat32, shape1);
                        output->apply_data();
                        tensor_element_add<float>(
                            tensor1->ptr<float>(), tensor2->ptr<float>(), output->ptr<float>(), tensor1->size());
                    }
                    else
                    {
                        LOG(INFO) << "Begin to broadcast tensor.";
                        auto broadcast_shape = broadcast_shapes(shape1, shape2);
                        tensor1->broadcast(broadcast_shape);
                        tensor2->broadcast(broadcast_shape);
                        tensor_element_add<float>(
                            tensor1->ptr<float>(), tensor2->ptr<float>(), output->ptr<float>(), tensor1->size());
                    }
                }
                else
                {
                    LOG(ERROR) << "Add operator needs two tensor operband.";
                }
            }
            else if (node->token_.type == TokenType::TokenMul)
            {
                if (std::holds_alternative<Tensor::TensorPtr>(left) && std::holds_alternative<Tensor::TensorPtr>(right))
                {
                    auto tensor1 = std::get<Tensor::TensorPtr>(left);
                    auto tensor2 = std::get<Tensor::TensorPtr>(right);
                    auto shape1 = tensor1->shapes();
                    auto shape2 = tensor2->shapes();
                    if (shape1 == shape2)
                    {
                        output = std::make_shared<Tensor>(DataType::DataTypeFloat32, shape1);
                        output->apply_data();
                        tensor_element_mul<float>(
                            tensor1->ptr<float>(), tensor2->ptr<float>(), output->ptr<float>(), tensor1->size());
                    }
                    else
                    {
                        LOG(INFO) << "Begin to broadcast tensor.";
                        auto broadcast_shape = broadcast_shapes(shape1, shape2);
                        tensor1->broadcast(broadcast_shape);
                        tensor2->broadcast(broadcast_shape);
                        tensor_element_mul<float>(
                            tensor1->ptr<float>(), tensor2->ptr<float>(), output->ptr<float>(), tensor1->size());
                    }
                }
                else
                {
                    LOG(INFO) << typeid(left).name() << " " << typeid(right).name();
                    LOG(ERROR) << "Mul operator needs two tensor operband.";
                }
            }
            else if (node->token_.type == TokenType::TokenDiv)
            {
                if (std::holds_alternative<Tensor::TensorPtr>(left) && std::holds_alternative<int>(right))
                {
                    auto tensor1 = std::get<Tensor::TensorPtr>(left);
                    auto int_value = std::get<int>(right);
                    auto shape1 = tensor1->shapes();
                    output = std::make_shared<Tensor>(DataType::DataTypeFloat32, shape1);
                    output->apply_data();
                    tensor_element_div<float>(tensor1->ptr<float>(), int_value, output->ptr<float>(), tensor1->size());
                }
                else
                {
                    LOG(ERROR) << "Div operator needs one tensor operband and one int value.";
                }
            }
            op_stack.push(output);
        }
        else if (node->token_.type == TokenType::TokenSqrt)
        {
            CHECK(op_stack.size() >= 1)
                << "Dqrt operator stack size is less than 2, but this operator needs 2 operband.";
            auto right = op_stack.top();
            op_stack.pop();
            using R_T = std::decay_t<decltype(right)>;
            Tensor::TensorPtr output = nullptr;
            if constexpr (std::is_same<R_T, Tensor::TensorPtr>::value)
            {
                auto tensor1 = std::get<Tensor::TensorPtr>(right);
                auto shape1 = tensor1->shapes();
                output = std::make_shared<Tensor>(DataType::DataTypeFloat32, shape1);
                output->apply_data();
                tensor_element_sqrt<float>(tensor1->ptr<float>(), output->ptr<float>(), tensor1->size());
            }
            else
            {
                LOG(ERROR) << "Div operator needs one tensor operband and one int value.";
            }
            op_stack.push(output);
        }
    }
    // 最后栈中只剩下一个元素，就是计算结果
    CHECK(op_stack.size() == 1) << "Operator stack size is not 1, check your expression.";
    auto result = std::move(op_stack.top());
    auto output = std::get<Tensor::TensorPtr>(result);
    outputs_[0] = output;
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx