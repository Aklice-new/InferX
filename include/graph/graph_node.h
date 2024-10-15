#ifndef _GRPAH_OP_H_
#define _GRPAH_OP_H_

#include <memory>

#include "graph/graph.h"
#include "graph/pnnx/ir.h"
#include "layer/layer.h"

namespace inferx
{
namespace graph
{

/**
 * @brief Graph Operator is a warpper for a layer, it consists of a layer and its input and output tensors information
 *        it is represented as a node in the graph
 */
using namespace inferx::layer;

class GraphEdge;

class GraphNode
{

    friend class Graph;
    friend class GraphEdge;

private:
    std::string name_{""};
    std::string type_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::shared_ptr<Layer> layer_;
    std::map<std::string, std::shared_ptr<GraphEdge>> in_edges_;
    std::vector<std::string> out_nodes_;
    size_t execute_time_ = -1;

    std::map<std::string, pnnx::Parameter> params_;
    std::map<std::string, pnnx::Attribute> attrs_;
};

} // namespace graph
} // namespace inferx

#endif