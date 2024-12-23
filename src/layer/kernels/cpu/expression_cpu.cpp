#include "core/common.h"
#include "core/tensor.h"
#include "layer/kernels/expression.h"
#include "utils/utils.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stack>
#include <type_traits>
#include <variant>
#include <algorithm>

#include <glog/logging.h>

namespace inferx
{
namespace layer
{

template <typename DType>
void tensor_element_add(DType* input1, DType* input2, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input1[i] + input2[i];
    }
}

template <typename DType>
void tensor_element_mul(DType* input1, DType* input2, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input1[i] * input2[i];
    }
}
template <typename DType>
void tensor_element_div(DType* input1, int divisor, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input1[i] / divisor;
    }
}
template <typename DType>
void tensor_element_sqrt(DType* input1, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = std::sqrt(input1[i]);
    }
}

template <typename DType>
struct binary_op_add
{
    DType operator()(const DType x, const DType y) const
    {
        return x + y;
    }
};
template <typename DType>
struct binary_op_sub
{
    DType operator()(const DType x, const DType y) const
    {
        return x - y;
    }
};
template <typename DType>
struct binary_op_mul
{
    DType operator()(const DType x, const DType y) const
    {
        return x * y;
    }
};
template <typename DType>
struct binary_op_div
{
    DType operator()(const DType x, const DType y) const
    {
        return x / y;
    }
};

template <typename BinaryOp>
static void tensor_elementwise_broadcast(
    std::vector<uint32_t>& broadcast_shape, const Tensor& tensor1, const Tensor& tensor2, Tensor::TensorPtr tensor_out)
{
    const BinaryOp op;
    auto dims = tensor1.shapes().size();
    CHECK(dims <= 4) << "Only support tensor with dims <= 4.";
    auto strides1 = tensor1.strides();
    auto strides2 = tensor2.strides();
    if (dims == 4)
    {
        auto N = broadcast_shape[0];
        auto C = broadcast_shape[1];
        auto H = broadcast_shape[2];
        auto W = broadcast_shape[3];

        for (size_t n = 0; n < N; n++)
        {
            for (size_t c = 0; c < C; c++)
            {
                for (size_t h = 0; h < H; h++)
                {
                    for (size_t w = 0; w < W; w++)
                    {
                        auto idx = n * C * H * W + c * H * W + h * W + w;
                        auto idx1 = n * strides1[0] + c * strides1[1] + h * strides1[2] + w * strides1[3];
                        auto idx2 = n * strides2[0] + c * strides2[1] + h * strides2[2] + w * strides2[3];
                        // const auto x = const_cast<const float*>(tensor1.const_ptr<float>());
                        tensor_out->ptr<float>()[idx]
                            = op(tensor1.const_ptr<float>()[idx1], tensor2.const_ptr<float>()[idx2]);
                    }
                }
            }
        }
    }
    else if (dims == 3)
    {
        auto C = broadcast_shape[0];
        auto H = broadcast_shape[1];
        auto W = broadcast_shape[2];
        for (size_t c = 0; c < C; c++)
        {
            for (size_t h = 0; h < H; h++)
            {
                for (size_t w = 0; w < W; w++)
                {
                    auto idx = c * H * W + h * W + w;
                    auto idx1 = c * strides1[0] + h * strides1[1] + w;
                    auto idx2 = c * strides2[0] + h * strides2[1] + w;
                    // const auto x = const_cast<const float*>(tensor1.const_ptr<float>());
                    tensor_out->ptr<float>()[idx]
                        = op(tensor1.const_ptr<float>()[idx1], tensor2.const_ptr<float>()[idx2]);
                }
            }
        }
    }
    else if (dims == 2)
    {
        auto H = broadcast_shape[0];
        auto W = broadcast_shape[1];
        for (size_t h = 0; h < H; h++)
        {
            for (size_t w = 0; w < W; w++)
            {
                auto idx = h * W + w;
                auto idx1 = h * strides1[0] + w;
                auto idx2 = h * strides2[0] + w;
                // const auto x = const_cast<const float*>(tensor1.const_ptr<float>());
                tensor_out->ptr<float>()[idx] = op(tensor1.const_ptr<float>()[idx1], tensor2.const_ptr<float>()[idx2]);
            }
        }
    }
    else if (dims == 1)
    {
        auto len = broadcast_shape[0];
        for (size_t l = 0; l < len; l++)
        {
            tensor_out->ptr<float>()[l] = op(tensor1.const_ptr<float>()[l], tensor2.const_ptr<float>()[l]);
        }
    }
}

/**
 * @brief 广播形状
 *        需要返回两个tensor广播完的形状
 *        两个tensor实际的形状
 *        每个维度是否被广播的信息
 */
std::vector<uint32_t> broadcast_shapes(const std::vector<uint32_t>& shapes1, const std::vector<uint32_t>& shapes2,
    std::vector<uint32_t>& broadcast_shape1, std::vector<uint32_t>& broadcast_shape2,
    std::vector<uint32_t>& is_broadcast1, std::vector<uint32_t>& is_broadcast2)
{
    // https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
    std::vector<uint32_t> broadcast_shape;
    auto size1 = shapes1.size();
    auto size2 = shapes2.size();
    auto max_size = std::max(size1, size2);
    broadcast_shape.resize(max_size, 0);
    broadcast_shape1.resize(max_size, 0);
    broadcast_shape2.resize(max_size, 0);
    is_broadcast1.resize(max_size, 0);
    is_broadcast2.resize(max_size, 0);

    std::vector<uint32_t> reverse_shape1, reverse_shape2;
    for (auto it = shapes1.rbegin(); it < shapes1.rend(); it++)
    {
        reverse_shape1.push_back(*it);
    }
    for (auto it = shapes2.rbegin(); it < shapes2.rend(); it++)
    {
        reverse_shape2.push_back(*it);
    }
    for (size_t i = 0; i < max_size; i++)
    {
        auto shape1 = i < size1 ? reverse_shape1[i] : 1;
        auto shape2 = i < size2 ? reverse_shape2[i] : 1;
        auto condition1 = shape1 != shape2 && (shape1 == 1 || shape2 == 1);
        auto condition2 = shape1 == shape2;
        CHECK(condition1 || condition2) << "Tensor dimenson fail to match.";
        auto shape = std::max(shape1, shape2);
        broadcast_shape[i] = shape;
        broadcast_shape1[i] = (i < size1 && reverse_shape1[i] >= shape) ? shape : 1; // 得到为了广播而reshape的维度
        is_broadcast1[i] = (i < size1 && shape1 >= shape) ? 0 : 1; // 表示当前维度是广播得到的
        broadcast_shape2[i] = (i < size2 && reverse_shape2[i] >= shape) ? shape : 1;
        is_broadcast2[i] = (i < size2 && shape2 >= shape) ? 0 : 1;
    }
    reverse(broadcast_shape.begin(), broadcast_shape.end());
    reverse(broadcast_shape1.begin(), broadcast_shape1.end());
    reverse(broadcast_shape2.begin(), broadcast_shape2.end());
    reverse(is_broadcast1.begin(), is_broadcast1.end());
    reverse(is_broadcast2.begin(), is_broadcast2.end());

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
                        // LOG(INFO) << "Begin to broadcast tensor.";
                        std::vector<uint32_t> broadcast_shape1, broadcast_shape2, is_broadcast1, is_broadcast2;
                        auto broadcast_shape = broadcast_shapes(
                            shape1, shape2, broadcast_shape1, broadcast_shape2, is_broadcast1, is_broadcast2);
                        auto b_tensor1 = tensor1->broadcast(broadcast_shape1, is_broadcast1);
                        auto b_tensor2 = tensor2->broadcast(broadcast_shape2, is_broadcast2);
                        output = std::make_shared<Tensor>(DataType::DataTypeFloat32, broadcast_shape);
                        output->apply_data();
                        tensor_elementwise_broadcast<binary_op_add<float>>(
                            broadcast_shape, b_tensor1, b_tensor2, output);
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
                        std::vector<uint32_t> broadcast_shape1, broadcast_shape2, is_broadcast1, is_broadcast2;
                        auto broadcast_shape = broadcast_shapes(
                            shape1, shape2, broadcast_shape1, broadcast_shape2, is_broadcast1, is_broadcast2);
                        auto b_tensor1 = tensor1->broadcast(broadcast_shape1, is_broadcast1);
                        auto b_tensor2 = tensor2->broadcast(broadcast_shape2, is_broadcast2);
                        output = std::make_shared<Tensor>(DataType::DataTypeFloat32, broadcast_shape);
                        output->apply_data();
                        tensor_elementwise_broadcast<binary_op_mul<float>>(
                            broadcast_shape, b_tensor1, b_tensor2, output);
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
                << "Sqrt operator stack size is less than 2, but this operator needs 2 operband.";
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
    outputs_[0]->copy_from(output->raw_ptr(), output->size());
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx