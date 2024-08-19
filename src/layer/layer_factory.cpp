#include "layer/layer_factory.h"
#include "layer/layer.h"
#include <memory>
#include <glog/logging.h>

namespace inferx
{

namespace layer
{
/* LayerRegister static member defination */

std::shared_ptr<LayerRegister::LayerCreateTable> LayerRegister::layer_table_ = nullptr;

// 返回LayerRegister的注册表，这里保证单例
std::shared_ptr<LayerRegister::LayerCreateTable> LayerRegister::get_table()
{
    if (LayerRegister::layer_table_ == nullptr)
    {
        LayerRegister::layer_table_ = std::make_shared<LayerCreateTable>();
        static LayerRegisterGarbageCollector c;
    }
    CHECK(layer_table_ != nullptr) << "Layer Register initialize failed!";
    return LayerRegister::layer_table_;
}

void LayerRegister::register_layer_creator(const std::string& layer_name, LayerCreateFunc& creator)
{
    CHECK(!layer_name.empty());
    CHECK(creator != nullptr);
    auto layer_tabel = get_table();
    CHECK_EQ(layer_tabel->count(layer_name), 0) << "Layer name: " << layer_name << " has already registered!";
    layer_tabel->insert({layer_name, creator});
}

std::shared_ptr<Layer> LayerRegister::create_layer(const std::string& layer_name)
{
    auto layer_table = get_table();
    LOG_IF(FATAL, layer_table->count(layer_name) == 0) << "Unknow layer type " << layer_name;

    const auto& creator = layer_table->at(layer_name);
    CHECK(creator != nullptr) << "Not find the crespond creator";
    auto layer_ptr = std::make_shared<Layer>(creator(layer_name));
    CHECK(layer_ptr != nullptr) << "Create layer " << layer_name << " failed . ";
    return layer_ptr;
}

std::vector<std::string> LayerRegister::get_registed_layers()
{
    std::vector<std::string> layer_names;
    auto layer_table = get_table();
    for (const auto& [name, creator] : *layer_table.get())
    {
        layer_names.push_back(name);
    }
    return layer_names;
}
} // namespace layer

} // namespace inferx