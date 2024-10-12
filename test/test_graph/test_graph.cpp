#include "graph/graph.h"

using namespace inferx::graph;
using namespace inferx::core;

int main()
{
    std::string param_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/pnnx_test/resnet18/res18.pnnx.param";
    std::string bin_path = "/home/aklice/WorkSpace/VSCode/Lab/InferX/test/pnnx_test/resnet18/res18.pnnx.bin";
    Graph graph(bin_path, param_path);
    // graph.load_model();
    // graph.infernce();
}