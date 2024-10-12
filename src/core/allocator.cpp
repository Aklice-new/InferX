#include "core/allocator.h"
#include "core/common.h"
#include "core/status.h"
#include "utils.h"

#include <cuda.h>
#include <glog/logging.h>

namespace inferx
{
namespace core
{

void Allocator::memcpy(void* dst, const void* src, size_t size, MemcpyKind kind, bool is_async)
{
    if (kind == MemcpyKind::HostToHost)
    {
        std::memcpy(dst, src, size);
    }
    else if (kind == MemcpyKind::HostToDevice)
    {
        // std::cout << "HostToDevice" << std::endl;
        // cudaStream_t stream = nullptr;
        // LOG(INFO) << "HostToDevice memory copy";
        // if (is_async)
        // {
        //     CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
        // }
        // else
        // {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
        // }
    }
    else if (kind == MemcpyKind::DeviceToHost)
    {
        // cudaStream_t stream = nullptr;
        // if (is_async)
        // {
        //     CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
        // }
        // else
        // {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
        // }
    }
    else if (kind == MemcpyKind::DeviceToDevice)
    {
        // cudaStream_t stream = nullptr;
        // if (is_async)
        // {
        //     CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
        // }
        // else
        // {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
        // }
    }
}

/**
 * @brief CPUAllocator
 *
 */
CPUAllocator::CPUAllocator()
{
    device_type_ = DeviceType::DeviceType_CPU;
}

void* CPUAllocator::allocate(size_t size)
{
    free_mutex.lock();
    auto it = m_free_memory.lower_bound(size);
    void* ptr = nullptr;
    if (it != m_free_memory.end())
    {
        ptr = it->second.back();
        it->second.pop_back();
        if (it->second.empty())
        {
            m_free_memory.erase(it);
        }
        free_mutex.unlock();

        alloc_mutex.lock();
        m_alloc_memory.insert({ptr, size});
        alloc_mutex.unlock();
        return ptr;
    }
    free_mutex.unlock();

    // new memory block
    ptr = aligned_alloc(size);
    alloc_mutex.lock();
    m_alloc_memory[ptr] = size;
    alloc_mutex.unlock();
    return ptr;
}

void* CPUAllocator::aligned_alloc(size_t size, size_t alignment)
{
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        throw new Status(StatusCode::Failed, "Failed to allocate memory");
    }
    return ptr;
}

void CPUAllocator::release(void* ptr)
{
    alloc_mutex.lock();
    if (m_alloc_memory.find(ptr) == m_alloc_memory.end())
    {
        throw new Status(StatusCode::Failed, "Failed to release memory, memory has not been allocated");
    }
    size_t size = m_alloc_memory.at(ptr);
    m_alloc_memory.erase(ptr);
    alloc_mutex.unlock();

    free_mutex.lock();
    m_free_memory[size].push_back(ptr);
    free_mutex.unlock();
}

DeviceType CPUAllocator::get_device_type()
{
    return device_type_;
}

CPUAllocator::~CPUAllocator()
{
    if (!m_alloc_memory.empty())
    {
        // spdlog::error("FATAL ERROR! Memory is still in use when CPUAllocator is destroyed");
        LOG(ERROR) << "FATAL ERROR! Memory is still in use when CPUAllocator is destroyed";
    }

    // free all need-free memory
    for (auto& it : m_free_memory)
    {
        for (auto& ptr : it.second)
        {
            ::free(ptr);
        }
    }
}

/**
 * @brief GPUAllocator
 *
 */
GPUAllocator::GPUAllocator(uint32_t device_id)
{
    device_type_ = DeviceType::DeviceType_GPU;
    device_id_ = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&(cuda_handle_.stream)));
    CUBLAS_CHECK(cublasCreate(&(cuda_handle_.cublas_handle)));
}

void* GPUAllocator::allocate(size_t size, bool is_async)
{
    free_mutex.lock();
    auto it = m_free_memory.lower_bound(size);
    void* ptr = nullptr;
    if (it != m_free_memory.end())
    {
        // 从待释放内存区域中找出来
        ptr = it->second.back();
        it->second.pop_back();
        if (it->second.empty())
        {
            m_free_memory.erase(it);
        }
        free_mutex.unlock();

        // 丢进已申请区域
        alloc_mutex.lock();
        m_alloc_memory[ptr] = size;
        alloc_mutex.unlock();
        return ptr;
    }
    free_mutex.unlock();

    // new memory block
    void* d_ptr = nullptr;
    if (is_async)
    {
        CUDA_CHECK(cudaMallocAsync(&d_ptr, size, cuda_handle_.stream));
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_ptr, size));
    }
    alloc_mutex.lock();
    m_alloc_memory[d_ptr] = size;
    alloc_mutex.unlock();
    return d_ptr;
}

void* GPUAllocator::allocate(size_t size)
{
    return allocate(size, false);
}

void GPUAllocator::release(void* ptr)
{
    alloc_mutex.lock();
    if (m_alloc_memory.find(ptr) == m_alloc_memory.end())
    {
        // spdlog::error("{} has not been allocated", ptr);
        LOG(ERROR) << "Failed to release memory, memory has not been allocated";
        // throw Status(StatusCode::Failed, "Failed to release memory, memory has not been allocated");
    }
    size_t size = m_alloc_memory[ptr];
    m_alloc_memory.erase(ptr);
    alloc_mutex.unlock();

    free_mutex.lock();
    m_free_memory[size].push_back(ptr);
    free_mutex.unlock();
}

DeviceType GPUAllocator::get_device_type()
{
    return device_type_;
}

void GPUAllocator::sync()
{
    CUDA_CHECK(cudaStreamSynchronize(cuda_handle_.stream));
}

GPUAllocator::~GPUAllocator()
{
    if (!m_alloc_memory.empty())
    {
        // spdlog::error("FATAL ERROR! Memory is still in use when GPUAllocator is destroyed");
        LOG(ERROR) << "FATAL ERROR! Memory is still in use when GPUAllocator is destroyed";
    }

    // free all need-free memory
    for (auto& it : m_free_memory)
    {
        for (auto& ptr : it.second)
        {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
}

/* CPU Factory static memeber define */

std::mutex CPUAllocatorFactory::mutex_;
std::shared_ptr<Allocator> CPUAllocatorFactory::allocator_ = nullptr;

/* GPU Factory static memeber define */

std::mutex GPUAllocatorFactory::mutex_;
std::unordered_map<uint32_t, std::shared_ptr<Allocator>> GPUAllocatorFactory::allocator_map_;

} // namespace core
} // namespace inferx