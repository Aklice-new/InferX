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

/**
 * @brief 根据逆波兰式计算表达式
 *
 */
StatusCode ExpressionLayer::forward_cpu()
{

    CHECK(parser_ != nullptr) << "Parser is null.";
    const auto inverse_polish_notation = parser_->GenerateSyntaxTree();
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
            using R_T = std::decay_t<decltype(right)>;
            using L_T = std::decay_t<decltype(left)>;

            Tensor::TensorPtr output = nullptr;

            if (node->token_.type == TokenType::TokenAdd)
            {
                if constexpr (std::is_same<L_T, Tensor::TensorPtr>::value
                    && std::is_same<R_T, Tensor::TensorPtr>::value)
                {
                    auto tensor1 = std::get<Tensor::TensorPtr>(left);
                    auto tensor2 = std::get<Tensor::TensorPtr>(right);
                    auto shape1 = tensor1->shapes();
                    auto shape2 = tensor2->shapes();
                    CHECK(shape1 == shape2) << "Add operator needs two tensor operband with same shape.";
                    output = std::make_shared<Tensor>(DataType::DataTypeFloat32, shape1);
                    tensor_element_add<float>(
                        tensor1->ptr<float>(), tensor2->ptr<float>(), output->ptr<float>(), tensor1->size());
                }
                else
                {
                    LOG(ERROR) << "Add operator needs two tensor operband.";
                }
            }
            else if (node->token_.type == TokenType::TokenMul)
            {
                if constexpr (std::is_same<L_T, Tensor::TensorPtr>::value
                    && std::is_same<R_T, Tensor::TensorPtr>::value)
                {
                    auto tensor1 = std::get<Tensor::TensorPtr>(left);
                    auto tensor2 = std::get<Tensor::TensorPtr>(right);
                    auto shape1 = tensor1->shapes();
                    auto shape2 = tensor2->shapes();
                    CHECK(shape1 == shape2) << "Add operator needs two tensor operband with same shape.";
                    output = std::make_shared<Tensor>(DataType::DataTypeFloat32, shape1);
                    tensor_element_mul<float>(
                        tensor1->ptr<float>(), tensor2->ptr<float>(), output->ptr<float>(), tensor1->size());
                }
                else
                {
                    LOG(ERROR) << "Mul operator needs two tensor operband.";
                }
            }
            else if (node->token_.type == TokenType::TokenDiv)
            {
                if constexpr (std::is_same<L_T, Tensor::TensorPtr>::value && std::is_same<R_T, int>::value)
                {
                    auto tensor1 = std::get<Tensor::TensorPtr>(left);
                    auto int_value = std::get<int>(right);
                    auto shape1 = tensor1->shapes();
                    output = std::make_shared<Tensor>(DataType::DataTypeFloat32, shape1);
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