#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <cstddef>
#include <memory>

#include "core/common.h"
#include "core/tensor.h"
#include "graph/pnnx/ir.h"
#include "layer/layer.h"

namespace inferx
{
namespace graph
{

using namespace inferx::core;
using namespace inferx::layer;

class GraphNode;
class GraphEdge;

class Graph
{
public:
    explicit Graph();
    explicit Graph(const std::string& bin_path, const std::string& param_path);

    StatusCode load_model();

    StatusCode load_model(const std::string& bin_path, const std::string& param_path);

    StatusCode infernce();

    StatusCode set_input(const Tensor& input_tensor);

private:
    void process_in_edges(const std::vector<pnnx::Operand*>& inputs, const std::shared_ptr<GraphNode>& graph_op);

    void process_out_edges(const std::vector<pnnx::Operand*>& outputs, const std::shared_ptr<GraphNode>& graph_op);

    void create_graph();

private:
    std::string bin_path_;
    std::string param_path_;
    std::unique_ptr<pnnx::Graph> graph_;
    size_t layer_nums_;
    size_t tensor_nums_;
    // std::map<std::string, size_t> name_to_index_map_; // 每个tensor的名字到整张图中tensors_的索引

private:
    std::vector<Tensor::TensorPtr> tensors_;
    std::map<std::string, Tensor::TensorPtr> tensors_map_;
    std::vector<std::shared_ptr<GraphNode>> layers_;
    std::map<std::string, std::shared_ptr<GraphNode>> layers_map_;
};
} // namespace graph
} // namespace inferx

#endif