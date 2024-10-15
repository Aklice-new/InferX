#include <cstddef>
#include <cstdint>
#include <glog/logging.h>
#include <memory>
#include <string>
#include <queue>
#include <algorithm>

#include "graph/graph.h"
#include "core/common.h"
#include "core/tensor.h"
#include "graph/graph_node.h"
#include "graph/graph_edge.h"
#include "graph/pnnx/ir.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace graph
{

Graph::Graph()
{
    graph_ = std::make_unique<pnnx::Graph>();
}

Graph::Graph(const std::string& bin_path, const std::string& param_path)
{
    graph_ = std::make_unique<pnnx::Graph>();
    bin_path_ = bin_path;
    param_path_ = param_path;
}

StatusCode Graph::load_model()
{
    CHECK(bin_path_.empty() == false) << "bin path is empty";
    CHECK(param_path_.empty() == false) << "param path is empty";
    // 通过pnnx::Graph的load函数加载模型
    int re = graph_->load(param_path_, bin_path_);
    if (re == -1)
    {
        LOG(ERROR) << "load model failed";
        return StatusCode::Failed;
    }
    // 获取网络模型的算子个数和tensor的个数
    layer_nums_ = graph_->ops.size();
    tensor_nums_ = graph_->operands.size();

    LOG(INFO) << "layer nums: " << layer_nums_;
    LOG(INFO) << "tensor nums: " << tensor_nums_;

    layers_.resize(layer_nums_);
    tensors_.resize(tensor_nums_);
    // initialize graph 将所有信息都从pnnx中读出来
    for (size_t i = 0; i < layer_nums_; i++)
    {
        // 将pnnx::Operator转换为GraphOperator
        pnnx::Operator* op = graph_->ops[i];
        std::shared_ptr<GraphNode> graph_op = std::make_shared<GraphNode>();
        graph_op->name_ = op->name;
        graph_op->type_ = op->type;
        graph_op->execute_time_ = -1;

        // 通过Operator的输入输出构建计算关系图
        process_in_edges(op->inputs, graph_op);

        process_out_edges(op->outputs, graph_op);

        graph_op->params_ = std::move(op->params);
        graph_op->attrs_ = std::move(op->attrs);

        layers_[i] = graph_op;
        layers_map_.insert({graph_op->name_, graph_op});
    }
    // 根据以上获取的信息，构建整个图的计算关系(重点：处理好各个layer之间对tensor的依赖关系)
    create_graph();
    return StatusCode::Success;
}

void Graph::process_in_edges(const std::vector<pnnx::Operand*>& inputs, const std::shared_ptr<GraphNode>& graph_op)
{
    for (const auto& input : inputs)
    {
        const pnnx::Operator* producer = input->producer;
        graph_op->inputs_.push_back(input->name); // 保存输入连接的节点
        std::vector<uint32_t> dims;
        for (auto dim : input->shape)
        {
            dims.push_back(static_cast<uint32_t>(dim));
        }
        auto in_edge = std::make_shared<GraphEdge>();
        in_edge->producer_name_ = producer->name;
        in_edge->name_ = input->name;
        in_edge->dims_ = dims;

        switch (input->type)
        {
        case 1: in_edge->dtype_ = DataType::DataTypeFloat32; break;
        case 2: in_edge->dtype_ = DataType::DataTypeFloat64; break;
        case 4: in_edge->dtype_ = DataType::DataTypeInt32; break;
        case 6: in_edge->dtype_ = DataType::DataTypeInt16; break;
        case 7: in_edge->dtype_ = DataType::DataTypeInt8; break;
        default: LOG(FATAL) << "unsupported data type"; break;
        }

        if (tensors_map_.find(input->name) == tensors_map_.end())
        {
            auto tensor = std::make_shared<Tensor>(input->name, in_edge->dtype_, dims);
            // LOG(INFO) << "create tensor: " << input->name;
            tensors_map_.insert({input->name, tensor});
            // tensors_.push_back(tensor);
        }
        in_edge->tensor_ = tensors_map_[input->name];

        graph_op->in_edges_.insert({producer->name, in_edge}); // 保存输入边
    }
}

void Graph::process_out_edges(const std::vector<pnnx::Operand*>& outputs, const std::shared_ptr<GraphNode>& graph_op)
{
    for (const auto& output : outputs)
    {
        graph_op->outputs_.push_back(output->name);
        const auto& consumers = output->consumers;
        for (const auto& consumer : consumers)
        {
            graph_op->out_nodes_.push_back(consumer->name); // 保存输出连接的节点
        }
    }
}

void Graph::create_graph()
{
    // 每个GraphEdge对应的就是一个input的tensor
    for (const auto& current_graph_node : this->layers_)
    {
        std::vector<Tensor::TensorPtr> input_tensors, output_tensors;
        // 1. 首先处理输入
        for (const auto& in_edge : current_graph_node->in_edges_)
        {
            input_tensors.push_back(in_edge.second->tensor_);
        }
        // 2. 然后处理输出
        for (const auto& out_edge : current_graph_node->outputs_)
        {
            if (tensors_map_.find(out_edge) == tensors_map_.end())
            {
                LOG(ERROR) << "ERROR in create_graph: " << out_edge << " not found";
            }
            output_tensors.push_back(tensors_map_[out_edge]);
        }
        if (current_graph_node->type_ == "pnnx.Input" || current_graph_node->type_ == "pnnx.Output")
        {
            continue;
        }
        // 实例化GraphNode中的Layer对象
        auto current_layer = LayerRegister::create_layer(current_graph_node->type_);
        // 3. 设置layer的输入输出
        current_layer->prepare_layer(input_tensors, output_tensors);
        // 4. 设置layer的 params and attrs
        if (current_graph_node->params_.empty() == false)
        {
            // LOG(INFO) << "load param: " << current_graph_node->name_;
            current_layer->load_param(current_graph_node->params_);
        }
        if (current_graph_node->attrs_.empty() == false)
        {
            // LOG(INFO) << "load model: " << current_graph_node->name_;
            current_layer->load_model(current_graph_node->attrs_);
        }
        current_graph_node->layer_ = current_layer;
    }
    // topsort the graph, 设置execute_time来表示每个节点的执行顺序
    size_t global_execute_time = 0;
    std::queue<std::shared_ptr<GraphNode>> node_queue;
    std::set<std::string> visited_node;
    for (const auto& [name_, _] : layers_map_)
    {
        // 如果这里有问题，可以改为通过名字来判断是否是输入节点
        if (_.get()->inputs_.empty())
        {
            LOG(INFO) << "push graph node : " << name_;
            node_queue.push(_);
        }
    }
    while (!node_queue.empty())
    {
        const auto& current_node = node_queue.front();
        node_queue.pop();
        current_node->execute_time_ = global_execute_time++;
        visited_node.insert(current_node->name_);
        for (const auto& out_node : current_node->out_nodes_)
        {
            auto next_node = layers_map_[out_node];
            bool is_ready = true;
            for (const auto& in_edge : next_node->in_edges_)
            {
                if (in_edge.second->producer_name_ == current_node->name_)
                {
                    continue;
                }
                if (visited_node.find(in_edge.second->producer_name_) == visited_node.end())
                {
                    is_ready = false;
                    break;
                }
            }
            if (is_ready)
            {
                node_queue.push(next_node);
            }
        }
    }

    std::sort(
        layers_.begin(), layers_.end(), [](const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b) {
            return a->execute_time_ < b->execute_time_;
        });

#ifdef DEBUG
    for (const auto& node : layers_)
    {
        LOG(INFO) << "layer name: " << node->name_ << " execute time: " << node->execute_time_;
    }
#endif
}

StatusCode Graph::load_model(const std::string& bin_path, const std::string& param_path)
{
    bin_path_ = bin_path;
    param_path_ = param_path;
    return load_model();
}

StatusCode Graph::infernce(Tensor& output_tensor)
{
    LOG(INFO) << "Begin to inference.\n Current graph has " << layers_.size() << " layers";

    for (const auto& node : layers_)
    {
        LOG(INFO) << "layer name: " << node->name_ << " execute time: " << node->execute_time_;
        if (node->type_ == "pnnx.Input" || node->type_ == "pnnx.Output")
        {
            continue;
        }
        node->layer_->forward();
    }
    output_tensor = tensors_map_[std::to_string(tensor_nums_ - 1)]->clone();
    return StatusCode::Success;
}

StatusCode Graph::set_input(Tensor& input_tensor)
{
    /**
     *  0号index的tensor是输入tensor
     */
    if (tensors_map_.find("0") == tensors_map_.end())
    {
        LOG(ERROR) << "input tensor not found";
        return StatusCode::Failed;
    }
    auto input = tensors_map_["0"];
    CHECK(input->shapes() == input_tensor.shapes()) << "input tensor dims not match";

    input->copy_from(input_tensor.raw_ptr(), static_cast<size_t>(input_tensor.size()));
    return StatusCode::Success;
}
} // namespace graph
} // namespace inferx