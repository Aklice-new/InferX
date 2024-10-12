#include "parser/parser.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <iostream>

#include <glog/logging.h>

namespace inferx
{
namespace parser
{

std::vector<std::shared_ptr<TokenNode>> ExpressionParser::GenerateSyntaxTree()
{
    LOG(INFO) << "Begin to GenerateSyntaxTree";
    std::vector<std::shared_ptr<TokenNode>> inverse_polish_notation;
    // tokenize
    Tokenize();
    // generate syntax tree
    uint32_t idx = 0;
    auto root = GenterateNextNode(idx);
    std::function<void(std::shared_ptr<TokenNode> node)> post_order_traversal = [&](std::shared_ptr<TokenNode> node) {
        if (node == nullptr)
        {
            return;
        }
        post_order_traversal(node->left);
        post_order_traversal(node->right);
        inverse_polish_notation.push_back(node);
    };
    // post order traversal
    CHECK(root != nullptr) << "Root is nullptr, check your expression, make sure it is valid.";
    post_order_traversal(root);
    return inverse_polish_notation;
}

void ExpressionParser::Tokenize()
{
    /*
    Expression中包含的所有的token如下：
    运算符： sqrt, div, add, mul
    括号： (, )
    运算数字： [0-9]+
    变量（输入）： @[0-9]+
    */
    LOG(INFO) << "Begin to Tokenize.";
    tokens_.clear();
    CHECK(!expression_.empty()) << "Expression is empty, please check your input.";

    expression_.erase(std::remove(expression_.begin(), expression_.end(), ' '), expression_.end());
    CHECK(!expression_.empty()) << "Expression is empty, please check your input.";

    for (uint32_t i = 0; i < expression_.size();)
    {
        char c = expression_.at(i);
        // check if it is add
        if (c == 'a')
        {
            CHECK(expression_.substr(i, 3) == "add")
                << "Invalid expression, illegal char " << i << "  please check your input.";
            Token token(TokenType::TokenAdd, i, i + 2, "add");
            tokens_.push_back(token);
            i += 3;
        }
        else if (c == 'm')
        {
            CHECK(expression_.substr(i, 3) == "mul")
                << "Invalid expression, illegal char " << i << "  please check your input.";
            Token token(TokenType::TokenMul, i, i + 2, "mul");
            tokens_.push_back(token);
            i += 3;
        }
        else if (c == 'd')
        {
            CHECK(expression_.substr(i, 3) == "div")
                << "Invalid expression, illegal char " << i << "  please check your input.";
            Token token(TokenType::TokenDiv, i, i + 2, "div");
            tokens_.push_back(token);
            i += 3;
        }
        else if (c == 's')
        {
            CHECK(expression_.substr(i, 4) == "sqrt")
                << "Invalid expression, illegal char " << i << "  please check your input.";
            Token token(TokenType::TokenSqrt, i, i + 3, "sqrt");
            tokens_.push_back(token);
            i += 4;
        }
        else if (c == '(')
        {
            Token token(TokenType::TokenLeftParen, i, i, "(");
            tokens_.push_back(token);
            i++;
        }
        else if (c == ')')
        {
            Token token(TokenType::TokenRightParen, i, i, ")");
            tokens_.push_back(token);
            i++;
        }
        else if (c == '@')
        {
            uint32_t j = i + 1;
            while (j < expression_.size() && expression_.at(j) >= '0' && expression_.at(j) <= '9')
            {
                j++;
            }
            Token token(TokenType::TokenTensor, i + 1, j - 1, expression_.substr(i + 1, j - i));
            tokens_.push_back(token);
            i = j;
        }
        else if (c >= '0' && c <= '9')
        {
            uint32_t j = i + 1;
            while (j < expression_.size() && expression_.at(j) >= '0' && expression_.at(j) <= '9')
            {
                j++;
            }
            Token token(TokenType::TokenNumber, i, j - 1, expression_.substr(i, j - i));
            tokens_.push_back(token);
            i = j;
        }
        else if (c == ',')
        {
            Token token(TokenType::TokenComma, i, i, ",");
            tokens_.push_back(token);
            i++;
        }
        else
        {
            LOG(FATAL) << "Invalid expression, illegal char " << i << "  please check your input.";
        }
    }
}

std::shared_ptr<TokenNode> ExpressionParser::GenterateNextNode(uint32_t& idx)
{
    if (idx >= tokens_.size())
    {
        return nullptr;
    }
    Token curr_token = tokens_[idx];
    auto curr_node = std::make_shared<TokenNode>(curr_token, nullptr, nullptr);

    if (curr_token.type == TokenType::TokenTensor || curr_token.type == TokenType::TokenNumber)
    {
        idx++;
        return curr_node;
    }
    else if (curr_token.type == TokenType::TokenSqrt)
    {
        idx++;
        CHECK(tokens_[idx].type == TokenType::TokenLeftParen) << "Sqrt must be followed by a left paren.";
        idx++;
        auto left = GenterateNextNode(idx);
        CHECK(tokens_[idx].type == TokenType::TokenRightParen) << "Sqrt must be followed by a right paren.";
        idx++;
        curr_node->left = left;
        return curr_node;
    }
    else if (curr_token.type == TokenType::TokenAdd || curr_token.type == TokenType::TokenMul
        || curr_token.type == TokenType::TokenDiv)
    {
        idx++;
        CHECK(tokens_[idx].type == TokenType::TokenLeftParen) << "Operator must be followed by a left paren.";
        idx++;
        auto left = GenterateNextNode(idx);
        CHECK(tokens_[idx].type == TokenType::TokenComma) << "Operator must be followed by a comma.";
        idx++;
        auto right = GenterateNextNode(idx);
        CHECK(tokens_[idx].type == TokenType::TokenRightParen) << "Operator must be followed by a right paren.";
        idx++;
        curr_node->left = left;
        curr_node->right = right;
        return curr_node;
    }
    else
    {
        LOG(FATAL) << "Invalid token type, please check your input.";
    }
    return nullptr;
}

const std::vector<Token> ExpressionParser::GetTokens()
{
    CHECK(!tokens_.empty()) << "Tokens is empty, please generate syntax tree firstly.";
    return tokens_;
}

} // namespace parser
} // namespace inferx