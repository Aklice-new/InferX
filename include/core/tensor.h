#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "core/allocator.h"
#include "core/common.h"
#include <cstddef>
#include <memory>
#include <atomic>
#include <vector>

namespace inferx
{
namespace core
{
/**
 * @brief Tensor
 *
 */
class Tensor
{
public:
    explicit Tensor();
    explicit Tensor(const std::string& name, DataType dtype, std::vector<size_t> shapes);
    explicit Tensor(
        DataType dtype, std::vector<size_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc = false);

    // 所有的构造函数都转发到这里完成
    void create(
        DataType dtype, std::vector<size_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc = false);

    DeviceType device_type() const;

    DataType dtype() const;

    size_t size() const;

    StatusCode to_cpu();

    StatusCode to_cuda();

    // Tensor(Tensor& tensor) = delete;
    Tensor& operator=(const Tensor& tensor);
    ~Tensor();

private:
    // static size_t dtype_to_bytes(DataType dtype);

    // ref_count ++
    void addref() const;
    // ref_count --
    void subref() const;
    // release tensor
    void release();

private:
    std::shared_ptr<std::atomic<int>> refcount_ptr_;
    //  引用计数, 表示该tensor中的data_ptr被几个Tensor对象持有，当data_ptr区未被初始化的时候为0

    void* data_ptr_{nullptr};
    DataType dtype_;
    size_t dims_; // dimensons of the tensor
    std::vector<size_t> m_shapes_;
    std::vector<size_t> m_strides_;
    std::shared_ptr<Allocator> allocator_;

public:
    using TensorPtr = std::shared_ptr<Tensor>;

    void* raw_ptr();

    void* gpu_data();

    void* cpu_data();

    template <typename T>
    T* ptr()
    {
        return static_cast<T*>(data_ptr_);
    }

    size_t byte_size();
};
} // namespace core
} // namespace inferx
#endif