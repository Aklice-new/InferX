#include "core/tensor.h"
#include "core/common.h"
#include "spdlog/spdlog.h"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>

namespace inferx
{
namespace core
{

static size_t dtype_to_bytes(DataType dtype)
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
        spdlog::error("Unknown data type.");
    }
    }
    return 0;
}

Tensor::Tensor()
{
    //   不需要做任何处理，因为没有data
}

Tensor::Tensor(DataType dtype, std::vector<size_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc)
{
    create(dtype, shapes, allocator, need_alloc);
}
void Tensor::create(DataType dtype, std::vector<size_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc)
{
    dtype_ = dtype;
    dims_ = shapes.size();
    m_shapes_ = shapes;
    m_strides_.resize(dims_);
    m_strides_[dims_ - 1] = 1;
    for (size_t i = 1; i < dims_; i++)
    {
        m_strides_[dims_ - 1 - i] = m_shapes_[dims_ - i] * m_strides_[dims_ - i];
    }
    // eg: shape[2, 3, 4] => strides[12, 4, 1] == [3*4, 4, 1]
    allocator_ = allocator;
    if (need_alloc)
    {
        data_ptr_ = allocator_->allocate(dtype_to_bytes(dtype_) * m_shapes_[0] * m_strides_[0]);
        refcount_ptr_ = std::make_shared<std::atomic<int>>(1);
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

    // 加上引用
    if (other.refcount_ptr_)
    {
        other.addref();
    }
    return *this;
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

    data_ptr_ = 0;
}

Tensor::~Tensor()
{
    release();
}

void* Tensor::raw_ptr()
{
    return data_ptr_;
}

template <typename T>
T* Tensor::ptr()
{
    return static_cast<T>(data_ptr_);
}

} // namespace core
} // namespace inferx