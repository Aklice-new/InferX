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
#include <memory>
#include <mutex>
#include <unordered_map>
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
    DeviceType_CPU,
    DeviceType_GPU,
    DeviceType_UNKNOWN
};

enum class MemcpyKind
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
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
    void memcpy(void* dst, const void* src, size_t size, MemcpyKind kind, bool is_async = false);
    virtual ~Allocator() {}
    DeviceType device_type_ = DeviceType::DeviceType_UNKNOWN;
    std::mutex alloc_mutex;
    std::mutex free_mutex;
    mutable std::unordered_map<void*, size_t> m_alloc_memory;
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

/**
 * @brief CPUAllocator 工厂，单例模式
 *        饿汉模式，加锁保证线程安全
 *
 */
class CPUAllocatorFactory
{
public:
    static std::shared_ptr<Allocator> get_instance()
    {
        mutex_.lock();
        if (allocator_ == nullptr)
        {
            allocator_ = std::make_shared<CPUAllocator>();
        }
        mutex_.unlock();
        return allocator_;
    }
    // static void delete_instance()
    // {
    //     if (allocator_ != nullptr)
    //     {
    //         allocator_.reset();
    //     }
    // }

private: // 禁止外部构造和拷贝
    CPUAllocatorFactory() = default;
    ~CPUAllocatorFactory() = default;
    CPUAllocatorFactory(const CPUAllocatorFactory&);
    const CPUAllocatorFactory& operator=(const CPUAllocatorFactory&);

private:
    static std::shared_ptr<Allocator> allocator_;
    static std::mutex mutex_;
};

/**
 * @brief GPUAllocator 工厂，多例模式
 *        饿汉模式，加锁保证线程安全
 *
 */
class GPUAllocatorFactory
{
public:
    static std::shared_ptr<Allocator> get_instance(uint32_t device_id = 0)
    {
        mutex_.lock();
        if (allocator_map_.count(device_id) == 0)
        {
            auto allocator_ptr = std::make_shared<GPUAllocator>(device_id);
            allocator_map_.emplace(device_id, allocator_ptr);
        }
        mutex_.unlock();
        return allocator_map_.at(device_id);
    }
    // static void delete_instance(uint32_t device_id = 0)
    // {
    //     if (allocator_ != nullptr)
    //     {
    //         allocator_.reset();
    //     }
    // }

private:
    GPUAllocatorFactory() = default;
    ~GPUAllocatorFactory() = default;
    GPUAllocatorFactory(const GPUAllocatorFactory&);
    const GPUAllocatorFactory& operator=(const GPUAllocatorFactory&);

private:
    static std::unordered_map<uint32_t, std::shared_ptr<Allocator>> allocator_map_;
    static std::mutex mutex_;
};

}; // namespace core
} // namespace inferx

#endif