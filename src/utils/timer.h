#include <chrono>
#include <vector>

namespace inferx
{
namespace utils
{
class Timer
{
public:
    Timer() = default;
    ~Timer() = default;
    void start() {}
    void stop();
    double get_time();
};
} // namespace utils
} // namespace inferx