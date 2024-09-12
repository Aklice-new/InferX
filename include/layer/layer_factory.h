#ifndef _LAYEFACTORY_H_
#define _LAYEFACTORY_H_

#include "layer/layer.h"
#include <map>
#include <memory>

namespace inferx
{

namespace layer
{

/**
 * @brief 算子注册中心，用于注册所有的算子，方便到时候直接调用
 *
 */
class LayerRegister
{
public:
    using LayerCreateFunc = layer::Layer* (*) (std::string layer_name);
    using LayerCreateTable = std::map<std::string, LayerCreateFunc>;

private:
    friend class LayerRegisterWrapper;
    friend class LayerRegisterGarbageCollector;

    // 保存着所有算子的名字和创建方式的映射
    static std::shared_ptr<LayerCreateTable> layer_table_;

public:
    static std::shared_ptr<LayerCreateTable> get_table();
    static void register_layer_creator(const std::string& layer_name, const LayerCreateFunc& creator);
    static std::shared_ptr<Layer> create_layer(const std::string& layer_name);
    static std::vector<std::string> get_registed_layers();
};

class LayerRegisterWrapper
{
public:
    explicit LayerRegisterWrapper(LayerRegister::LayerCreateFunc layer_creator_func, const std::string& layer_name)
    {
        LayerRegister::register_layer_creator(layer_name, layer_creator_func);
    }
    template <typename... Ts>
    explicit LayerRegisterWrapper(
        const LayerRegister::LayerCreateFunc& layer_creator_func, const std::string layer_name, const Ts&... Args)
        : LayerRegisterWrapper(layer_creator_func, Args...)
    {
        LayerRegister::register_layer_creator(layer_name, layer_creator_func);
    }
};

class LayerRegisterGarbageCollector
{
public:
    ~LayerRegisterGarbageCollector()
    {
        if (LayerRegister::layer_table_ != nullptr)
        {
            LayerRegister::layer_table_.reset();
            LayerRegister::layer_table_ = nullptr;
        }
    }
};

} // namespace layer
} // namespace inferx

#endif