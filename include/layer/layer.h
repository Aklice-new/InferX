#ifndef _LAYER_H_
#define _LAYER_H_
#include "core/allocator.h"
#include "core/common.h"
#include "core/tensor.h"

#include <string>
#include <vector>

/**
 * @brief Layer 是各个算子的抽象类
 *        forward函数用于转发计算过程到 cpu或者gpu上
 *        查看了其他如MNN，ncnn等框架的实现，将设备抽象为backend，
 *        然后单独为该backend上的算子实现一个对应的Layer，
 *        分开去管理不同backend上Layer的参数。
 *        这里不选择这种设计的方式，因为是以学习的目的，注重算子内部的实现，而不怎么注重多平台的兼容性
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

    /**
     * @brief 检查进行计算的合法性，同时根据layer device_type检查Tensor的位置，最后转发算子的具体操作到cpu or gpu
     *
     * @return StatusCode
     */
    virtual StatusCode forward();

    virtual StatusCode forward_gpu();
    virtual StatusCode forward_cpu();

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