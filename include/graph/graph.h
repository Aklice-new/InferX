#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "core/common.h"
#include "layer/layer.h"
#include "core/tensor.h"
#include "graph/pnnx/ir.h"
#include <cstddef>
#include <memory>
namespace inferx
{
namespace graph
{
using namespace inferx::core;
using namespace inferx::layer;
class Graph
{
public:
    explicit Graph();
    explicit Graph(const std::string& bin_path, const std::string& param_path)
        : bin_path_(bin_path)
        , param_path_(param_path)
    {
        graph_ = std::make_unique<pnnx::Graph>();
    }

    StatusCode load_model();

    StatusCode infernce();

private:
    std::string bin_path_;
    std::string param_path_;
    std::unique_ptr<pnnx::Graph> graph_;
    size_t layer_nums_;
    size_t tensor_nums_;

private:
    std::vector<Tensor> tensors_;
    std::vector<Layer> layers_;
};
} // namespace graph
} // namespace inferx

#endif