#ifndef _LAYER_H_
#define _LAYER_H_
#include "core/allocator.h"
#include "core/common.h"
#include "core/tensor.h"
#include "graph/pnnx/ir.h"

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
    explicit Layer() = default;

    explicit Layer(std::string layer_name)
        : layer_name_(layer_name)
    {
    }
    virtual ~Layer() = default;
    /**
     * @brief 检查进行计算的合法性，同时根据layer device_type检查Tensor的位置，最后转发算子的具体操作到cpu or gpu
     *
     * @return StatusCode
     */
    virtual StatusCode forward()
    {
        if (device_type_ == DeviceType::DeviceType_GPU)
        {
            // this->to_cuda();
            return forward_gpu();
        }
        else
        {
            // this->to_cpu();
            return forward_cpu();
        }
    }

    virtual StatusCode forward_gpu()
    {
        return StatusCode::NotImplemented;
    };
    virtual StatusCode forward_cpu()
    {
        return StatusCode::NotImplemented;
    };

    /**
     * @brief load layer specific parameter, usually some single param, not a Tensor.
     *
     * @return StatusCode
     */
    virtual StatusCode load_param(const std::map<std::string, pnnx::Parameter>& params)
    {
        return StatusCode::NotImplemented;
    };
    /**
     * @brief load layer specific weight data
     *
     * @return StatusCode
     */
    virtual StatusCode load_model(const std::map<std::string, pnnx::Attribute>& attributes)
    {
        return StatusCode::NotImplemented;
    }

    /**
     * @brief prepare layer info, such as input/output tensor shape, weight shape
     *
     * @return StatusCode
     */
    virtual StatusCode prepare_layer(
        const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs)
    {
        return StatusCode::NotImplemented;
    }
    /**
     * @brief move this layer to CPU
     *
     * @return StatusCode
     */
    StatusCode to_cpu()
    {
        return StatusCode::NotImplemented;
    }
    /**
     * @brief move this layer to GPU
     *
     * @return StatusCode
     */
    StatusCode to_cuda()
    {
        return StatusCode::NotImplemented;
    }

    DeviceType device_type() const
    {
        return device_type_;
    }

    std::string layer_type() const
    {
        return layer_name_;
    }

protected:
    DeviceType device_type_{DeviceType::DeviceType_UNKNOWN};
    std::string layer_name_;
    // std::vector<Tensor> weights_; // include all tensor param
    //                               // all non-tensor param should be declare in specified layer
    std::vector<Tensor::TensorPtr> inputs_;
    std::vector<Tensor::TensorPtr> outputs_;
};
} // namespace layer

} // namespace inferx

#endif