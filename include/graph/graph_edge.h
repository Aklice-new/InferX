#ifndef _GRAPH_EDGE_H_
#define _GRAPH_EDGE_H_

#include <cstddef>
#include <string>

#include "core/common.h"
#include "core/tensor.h"
#include "graph/graph.h"
namespace inferx
{
namespace graph
{

class GraphEdge
{
    friend class Graph;
    friend class GraphNode;

public:
    std::string producer_name_;
    std::string name_;
    Tensor::TensorPtr tensor_;
    std::vector<uint32_t> dims_;
    DataType dtype_;
};
} // namespace graph
} // namespace inferx

#endif