/**
 * @file parser.h
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-27
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _PARSER_H_
#define _PARSER_H_

#include <cstdint>
#include <string>
#include <memory>

namespace inferx
{
namespace parser
{
/**
 * @brief every token type
 *
 */
enum class TokenType
{
    TokenUnknown = -1,
    TokenNumber = 0,
    TokenAdd = 1,
    TokenSub = 2,
    TokenMul = 3,
    TokenDiv = 4,
    TokenLeftParen = 5,
    TokenRightParen = 6,
    TokenStop = 7,
};
/**
 * @brief every token information, include type, start, end, value
 *
 */
struct Token
{
    TokenType type = TokenType::TokenUnknown;
    uint32_t start;
    uint32_t end;
    std::string value;
    Token() = default;
    Token(TokenType type, uint32_t start, uint32_t end, std::string value)
        : type(type)
        , start(start)
        , end(end)
        , value(value)
    {
    }
};

/**
 * @brief Token Node in the syntax tree
 *
 */
struct TokenNode
{
    Token token_;
    uint32_t tensor_idx_;
    std::shared_ptr<TokenNode> left = nullptr;
    std::shared_ptr<TokenNode> right = nullptr;
    TokenNode() = default;
    TokenNode(Token token, uint32_t tensor_idx, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right)
        : token_(token)
        , tensor_idx_(tensor_idx)
        , left(left)
        , right(right){};
};

} // namespace parser
} // namespace inferx

#endif