#include "core/allocator.h"
#include "core/status.h"
#include "core/tensor.h"
#include "spdlog/spdlog.h"
#include "utils/utils.h"
#include <cstdio>
#include <memory>

int main()
{
    auto gpu_allocator = std::make_shared<inferx::core::GPUAllocator>();

    for (int i = 0; i < 100000; i++)
    {
        spdlog::info("{} time allocate memory!", i);
        {
            inferx::Timer timer;
            inferx::core::Tensor tensor_a(inferx::core::DataTypeInt32, {1000, 1000, 1000}, gpu_allocator, true);
            spdlog::info("apply memory cost {}", timer.get_time());
        }
    }

    return 0;
}