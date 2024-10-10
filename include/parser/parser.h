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
#include <vector>

namespace inferx
{
namespace parser
{
/**
 * @brief every token type
 *
 */
enum class TokenType : int
{
    TokenUnknown = -1,
    TokenTensor,
    TokenNumber,
    TokenAdd,
    TokenMul,
    TokenDiv,
    TokenSqrt,
    TokenLeftParen,
    TokenRightParen,
    TokenComma,
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
    std::shared_ptr<TokenNode> left = nullptr;
    std::shared_ptr<TokenNode> right = nullptr;
    TokenNode() = default;
    TokenNode(Token token, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right)
        : token_(token)
        , left(left)
        , right(right){};
};

/**
 * @brief Expression Parser
 *   include lexer, parser, syntax tree
 */
class ExpressionParser
{
public:
    explicit ExpressionParser(std::string expression)
        : expression_(std::move(expression)){};

    /**
     * @brief 对词法分析的结果进行语法分析，然后生成语法树，再根据语法树的后序遍历生成逆波兰表达式
     *      https://oi-wiki.org/misc/expression/
     */
    std::vector<std::shared_ptr<TokenNode>> GenerateSyntaxTree();

    /**
     * @brief return the tokenized tokens
     *
     */

    const std::vector<Token> GetTokens();

private:
    /**
     * @brief 对表达式进行词法分析
     *
     */
    void Tokenize();
    /**
     * @brief 递归的生成语法分析树
     *
     */
    std::shared_ptr<TokenNode> GenterateNextNode(uint32_t& idx);

    std::vector<Token> tokens_;
    std::string expression_;
};

} // namespace parser
} // namespace inferx

#endif