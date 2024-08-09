#ifndef _STATUS_H_
#define _STATUS_H_

#include "common.h"

#include <exception>
#include <string>

namespace inferx
{
namespace core
{
// enum class StatusCode;
class Status : public std::exception
{
public:
    Status(int code = StatusCode::Success, std::string err_message = "")
        : code_(code)
        , message_(err_message)
    {
    }

    Status(const Status& other) = default;

    Status& operator=(const Status& other) = default;

    Status& operator=(int code)
    {
        this->code_ = code;
        return *this;
    }

    bool operator==(int code) const
    {
        return code_ == code;
    };

    bool operator!=(int code) const
    {
        return code_ != code;
    };

    operator int() const
    {
        return code_;
    };

    int32_t get_err_code() const
    {
        return code_;
    };

    const std::string& get_err_msg() const
    {
        return message_;
    };

    void set_err_msg(const std::string& err_msg)
    {
        message_ = err_msg;
    };

    const char* what() const noexcept override
    {
        return message_.c_str();
    };

private:
    int code_ = StatusCode::Success;
    std::string message_;
};
} // namespace core
} // namespace inferx

#endif