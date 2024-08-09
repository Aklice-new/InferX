#ifndef _ALLOCATOR_H_
#define _ALLOCATOR_H_
/**
 * @file allocator.h
 * @author aklice
 * @brief Allocator is a abstract class for memory allocation.
 *        one should implement the allocate and release function on different platform.
 * @version 0.1
 * @date 2024-08-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/nonecopyable.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace inferx
{
namespace core
{

class Buffer;

enum class DeviceType
{
    CPUALLOCATOR,
    GPUALLOCATOR
};

/**
 * @brief 内存分配器的设计思路是如下：
        如果当前有空闲的内存块，返回一个大于等于需要申请的空间的内存块的地址；
        如果没有空闲的内存块，申请一个新的内存块，返回这个内存块的地址。
        内存块的管理是通过一个map来管理的，key是内存块的地址，value是内存块的大小。
 *
 */
class Allocator : public NoneCopyable
{
public:
    explicit Allocator() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void release(void* ptr) = 0;
    virtual DeviceType get_device_type() = 0;
    virtual ~Allocator() {}
    DeviceType device_type_;
    std::mutex alloc_mutex;
    std::mutex free_mutex;
    mutable std::map<void*, size_t> m_alloc_memory;
    mutable std::map<size_t, std::vector<void*>> m_free_memory;
};

class CPUAllocator : public Allocator
{
public:
    CPUAllocator();
    virtual void* allocate(size_t size) override;
    virtual void release(void* ptr) override;
    virtual DeviceType get_device_type() override;
    ~CPUAllocator();

private:
    DeviceType device_type_ = DeviceType::CPUALLOCATOR;
    void* aligned_alloc(size_t size, size_t alignment = 32);
};

class GPUAllocator : public Allocator
{
public:
    struct CudaHandle
    {
        cudaStream_t stream{nullptr};
        cublasHandle_t cublas_handle{nullptr};
    };
    GPUAllocator(uint32_t device_id = 0);
    virtual void* allocate(size_t size) override;
    void* allocate(size_t size, bool is_async);
    virtual void release(void* ptr) override;
    virtual DeviceType get_device_type() override;
    void sync();
    ~GPUAllocator();

private:
    uint32_t device_id_;
    CudaHandle cuda_handle_;
};

}; // namespace core
} // namespace inferx

#endif