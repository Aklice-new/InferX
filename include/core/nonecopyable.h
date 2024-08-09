#ifndef _NONECOPYABLE_H_
#define _NONECOPYABLE_H_

/**
 * @brief None Copyable is a class that can not be copied or assigned.
 *
 */
namespace inferx
{
namespace core
{
class NoneCopyable
{
public:
    NoneCopyable() = default;
    NoneCopyable(const NoneCopyable&) = delete;
    NoneCopyable& operator=(const NoneCopyable&) = delete;
    virtual ~NoneCopyable() = default;
};

} // namespace core
} // namespace inferx
#endif