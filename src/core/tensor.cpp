#include "core/tensor.h"
#include "core/allocator.h"
#include "core/common.h"
#include <cstddef>
#include <cstdint>
#include <glog/logging.h>
#include <atomic>
#include <cassert>
#include <memory>

namespace inferx
{
namespace core
{

static uint32_t dtype_to_bytes(DataType dtype)
{
    switch (dtype)
    {
    case DataTypeFloat32: return 4;
    case DataTypeFloat64: return 8;
    case DataTypeInt32: return 4;
    case DataTypeInt16: return 2;
    case DataTypeInt8: return 1;
    default:
    {
        LOG(ERROR) << "Unknown data type";
    }
    }
    return 0;
}

Tensor::Tensor()
{
    //   不需要做任何处理，因为没有data
}

Tensor::Tensor(const std::string& name, DataType dtype, std::vector<uint32_t> shapes)
{
    create(dtype, shapes, CPUAllocatorFactory::get_instance(), false);
}

Tensor::Tensor(DataType dtype, std::vector<uint32_t> shapes)
{
    create(dtype, shapes, CPUAllocatorFactory::get_instance(), false);
}

Tensor::Tensor(DataType dtype, std::vector<uint32_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc)
{
    if (allocator == nullptr)
    {
        allocator = CPUAllocatorFactory::get_instance();
    }
    create(dtype, shapes, allocator, need_alloc);
}
void Tensor::create(DataType dtype, std::vector<uint32_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc)
{
    dtype_ = dtype;
    dims_ = shapes.size();
    m_shapes_ = shapes;
    m_strides_.resize(dims_);
    m_strides_[dims_ - 1] = 1;
    for (uint32_t i = 1; i < dims_; i++)
    {
        m_strides_[dims_ - 1 - i] = m_shapes_[dims_ - i] * m_strides_[dims_ - i];
    }
    // eg: shape[1, 2, 3, 4] => strides[24, 12, 4, 1] == [2 * 3 * 4, 3 * 4, 4, 1]
    allocator_ = allocator;
    if (need_alloc)
    {
        data_ptr_ = allocator_->allocate(static_cast<size_t>(byte_size()));
        refcount_ptr_ = std::make_shared<std::atomic<int>>(1);
    }
}

void Tensor::apply_data(std::shared_ptr<Allocator> allocator)
{
    if (data_ptr_ == nullptr)
    {
        data_ptr_ = allocator->allocate(static_cast<size_t>(byte_size()));
        refcount_ptr_ = std::make_shared<std::atomic<int>>(1);
    }
    else
    {
        LOG(INFO) << "Already applied data. Don't try to allocate again.";
    }
}

// Tensor::Tensor(Tensor& tensor)
// {
//     addref();
// }

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this == &other)
    {
        return *this;
    }

    // 首先 释放当前的tensor的内容
    this->release();

    // 再将另一个tensor中的成员拷贝进来
    this->dims_ = other.dims_;
    this->refcount_ptr_ = other.refcount_ptr_;
    this->allocator_ = other.allocator_;
    this->m_strides_ = other.m_strides_;
    this->m_shapes_ = other.m_shapes_;
    this->dtype_ = other.dtype_;
    this->data_ptr_ = other.data_ptr_;

    // 加上引用
    if (other.refcount_ptr_)
    {
        other.addref();
    }
    return *this;
}

Tensor Tensor::clone()
{
    Tensor newTensor = *this;
    newTensor.data_ptr_ = allocator_->allocate(static_cast<size_t>(byte_size()));
    newTensor.refcount_ptr_ = std::make_shared<std::atomic<int>>(1);
    newTensor.copy_from(this->data_ptr_, size());
    return newTensor;
}

void Tensor::addref() const
{
    refcount_ptr_->fetch_add(1, std::memory_order_acq_rel);
}

void Tensor::subref() const
{
    refcount_ptr_->fetch_sub(1, std::memory_order_acq_rel);
}

void Tensor::release()
{
    if (refcount_ptr_)
    {
        subref();
        if (*refcount_ptr_ == 0)
        {
            allocator_->release(data_ptr_);
        }
        // refcount_ptr_.reset(); 智能指针可以自动释放，所以这里应该不用。
    }

    data_ptr_ = nullptr;
}

Tensor::~Tensor()
{
    release();
}

void* Tensor::raw_ptr()
{
    return data_ptr_;
}
#ifdef ENABLE_CUDA
void* Tensor::gpu_data()
{
    if (device_type() == DeviceType::DeviceType_GPU)
    {
        return data_ptr_;
    }
    else
    {
        this->to_cuda();
        return data_ptr_;
    }
}
#endif
void* Tensor::cpu_data()
{
    if (device_type() == DeviceType::DeviceType_CPU)
    {
        return data_ptr_;
    }
    else
    {
        this->to_cpu();
        return data_ptr_;
    }
}

void Tensor::copy_from(const void* src, uint32_t size)
{
    if (this->data_ptr_ == nullptr)
    {
        apply_data(allocator_);
    }

    if (this->device_type() == DeviceType::DeviceType_CPU)
    {
        allocator_->memcpy(data_ptr_, src, static_cast<size_t>(size * dtype_to_bytes(dtype_)), MemcpyKind::HostToHost);
    }
    else if (this->device_type() == DeviceType::DeviceType_GPU)
    {
        allocator_->memcpy(
            data_ptr_, src, static_cast<size_t>(size * dtype_to_bytes(dtype_)), MemcpyKind::DeviceToHost);
    }
    else
    {
        LOG(ERROR) << "Unknown allocator type : " << static_cast<int>(allocator_->get_device_type());
        LOG(ERROR) << "Unknown device type : " << static_cast<int>(device_type());
    }
}

Tensor Tensor::reshape(std::vector<uint32_t> dims)
{
    Tensor newTensor = *this;
    newTensor.m_shapes_ = dims;
    newTensor.m_strides_.resize(dims_);
    newTensor.m_strides_[dims_ - 1] = 1;
    for (uint32_t i = 1; i < dims_; i++)
    {
        newTensor.m_strides_[dims_ - 1 - i] = newTensor.m_shapes_[dims_ - i] * newTensor.m_strides_[dims_ - i];
    }
    return std::move(newTensor);
}

void Tensor::Reshape(std::vector<uint32_t> dims)
{
    m_shapes_ = dims;
    m_strides_.resize(dims_);
    m_strides_[dims_ - 1] = 1;
    for (uint32_t i = 1; i < dims_; i++)
    {
        m_strides_[dims_ - 1 - i] = m_shapes_[dims_ - i] * m_strides_[dims_ - i];
    }
}

/* 模版的实现最好和声明放在一起，否则会出现链接错误
template <typename T>
T* Tensor::ptr()
{
    return static_cast<T>(data_ptr_);
}
*/
uint32_t Tensor::byte_size()
{
    return dtype_to_bytes(dtype_) * m_shapes_[0] * m_strides_[0];
}

Tensor Tensor::broadcast(std::vector<uint32_t> real_shape, std::vector<uint32_t> is_broadcast)
{
    // TODO : broadcast
    const auto& old_shapes = shapes();
    uint32_t new_size = 1;
    for (auto shape : real_shape)
    {
        new_size *= shape;
    }
    CHECK(new_size == size()) << "Broadcast error, the size of two tensor is different, can't do broadcast.";
    Tensor new_tensor = *this;
    // 在上面这个拷贝构造过程中关键的属性都已经完成了复制
    // 重要的是m_shapes_是广播后的shape，同时m_strides_有所不同
    new_tensor.Reshape(real_shape);
    // 将广播出来的那些维度的strides数值置为0
    for (int i = 0; i < dims_; i++)
    {
        new_tensor.m_strides_[i] *= (1 ^ is_broadcast[i]);
    }

    return std::move(new_tensor);
}

DeviceType Tensor::device_type() const
{
    return this->allocator_->device_type_;
}

std::shared_ptr<Allocator> Tensor::allocator() const
{
    return allocator_;
}

DataType Tensor::dtype() const
{
    return dtype_;
}

uint32_t Tensor::size() const
{
    return m_shapes_[0] * m_strides_[0];
}

std::vector<uint32_t> Tensor::shapes() const
{
    return m_shapes_;
}

std::vector<uint32_t> Tensor::strides() const
{
    return m_strides_;
}

StatusCode Tensor::to_cpu()
{
    if (device_type() == DeviceType::DeviceType_CPU)
    {
        return StatusCode::Success;
    }
    const DeviceType on_where = device_type();
    if (on_where == DeviceType::DeviceType_GPU)
    {
        uint32_t size = byte_size();
        void* cpu_data_ptr = nullptr;
        auto cpu_allocator_instance = CPUAllocatorFactory::get_instance();
        cpu_data_ptr = cpu_allocator_instance->allocate(static_cast<size_t>(size));
        cpu_allocator_instance->memcpy(cpu_data_ptr, data_ptr_, static_cast<size_t>(size), MemcpyKind::HostToDevice);
        allocator_->release(data_ptr_); // 释放GPU上的内存
        data_ptr_ = cpu_data_ptr;
        allocator_ = cpu_allocator_instance;
        return StatusCode::Success;
    }
    else if (on_where == DeviceType::DeviceType_CPU)
    {
        LOG(INFO) << "Tensor is already on CPU.";
        return StatusCode::Success;
    }
    return StatusCode::Failed;
}

#ifdef ENABLE_CUDA
StatusCode Tensor::to_cuda()
{
    if (device_type() == DeviceType::DeviceType_GPU)
    {
        return StatusCode::Success;
    }
    const DeviceType on_where = device_type();
    if (on_where == DeviceType::DeviceType_CPU)
    {
        uint32_t size = byte_size();
        void* cuda_data_ptr = nullptr;
        auto gpu_allocator_instance = GPUAllocatorFactory::get_instance();
        cuda_data_ptr = gpu_allocator_instance->allocate(static_cast<size_t>(size));
        gpu_allocator_instance->memcpy(cuda_data_ptr, data_ptr_, static_cast<size_t>(size), MemcpyKind::HostToDevice);
        allocator_->release(data_ptr_); // 释放CPU上的内存
        data_ptr_ = cuda_data_ptr;
        allocator_ = gpu_allocator_instance;
        return StatusCode::Success;
    }
    else if (on_where == DeviceType::DeviceType_GPU)
    {
        LOG(INFO) << "Tensor is already on GPU.";
        return StatusCode::Success;
    }
    return StatusCode::Failed;
}
#endif
} // namespace core
} // namespace inferx