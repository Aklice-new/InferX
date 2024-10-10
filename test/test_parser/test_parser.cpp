#include <iostream>

#include "parser/parser.h"

#include <glog/logging.h>

using namespace inferx::parser;
int main()
{

    std::string expression = "add(mul(@1, @2),mul(@3, sqrt(@4)))";
    ExpressionParser parser(expression);

    auto inverse_polish_notation = parser.GenerateSyntaxTree();
    auto tokens = parser.GetTokens();
    LOG(INFO) << "Tokens:";
    for (auto token : tokens)
    {
        std::cout << token.value << std::endl;
    }

    LOG(INFO) << "Inverse Polish Notation :";
    for (auto& node : inverse_polish_notation)
    {
        std::cout << node->token_.value << std::endl;
    }
}