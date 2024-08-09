#ifndef _LAYER_H_
#define _LAYER_H_
#include "core/allocator.h"
#include "core/common.h"
#include "core/tensor.h"

#include <string>
#include <vector>

/**
 * @brief Layer 是各个算子的抽象类，forward函数执行具体的计算过程
 *        layer 的数据区域保存该算子需要的权重data
 */
namespace inferx
{
namespace layer
{
using namespace inferx::core;
class Layer
{
public:
    Layer() = delete;

    explicit Layer(std::string layer_name);

    // virtual StatusCode forward(const std::vector<Tensor::TensorPtr> bottoms, std::vector<Tensor::TensorPtr> tops)
    // const;

    // virtual StatusCode forward_inplace(std::vector<Tensor::TensorPtr> bottoms) const;

    virtual StatusCode forward();

    /**
     * @brief load layer specific parameter, usually some single param, not a Tensor.
     *
     * @return StatusCode
     */
    virtual StatusCode load_param();
    /**
     * @brief load layer specific weight data
     *
     * @return StatusCode
     */
    virtual StatusCode load_model();

    /**
     * @brief allocate tensor memory for weights
     *
     * @return StatusCode
     */
    virtual StatusCode prepare_weight();
    /**
     * @brief move this layer to GPU
     *
     * @return StatusCode
     */
    virtual StatusCode to_cuda();

private:
    DeviceType device_type_;
    std::string layer_name_;
    std::vector<Tensor> weights_; // include all tensor param
                                  // all non-tensor param should be declare in specified layer
    std::vector<Tensor::TensorPtr> inputs_;
    std::vector<Tensor::TensorPtr> outputs_;
};
} // namespace layer

} // namespace inferx

#endif