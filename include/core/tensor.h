#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "core/allocator.h"
#include "core/common.h"
#include <cstddef>
#include <cstdint>
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
    explicit Tensor(const std::string& name, DataType dtype, std::vector<uint32_t> shapes);
    explicit Tensor(DataType dtype, std::vector<uint32_t> shapes);
    explicit Tensor(
        DataType dtype, std::vector<uint32_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc = false);

    // 所有的构造函数都转发到这里完成
    void create(
        DataType dtype, std::vector<uint32_t> shapes, std::shared_ptr<Allocator> allocator, bool need_alloc = false);

    void apply_data(std::shared_ptr<Allocator> allocator = CPUAllocatorFactory::get_instance());

    DeviceType device_type() const;

    std::shared_ptr<Allocator> allocator() const;

    DataType dtype() const;

    uint32_t size() const;

    std::vector<uint32_t> shapes() const;

    StatusCode to_cpu();

    StatusCode to_cuda();

    // Tensor(Tensor& tensor) = delete;
    Tensor& operator=(const Tensor& tensor);
    Tensor clone();
    ~Tensor();

private:
    // static uint32_t dtype_to_bytes(DataType dtype);

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
    uint32_t dims_; // dimensons of the tensor
    std::vector<uint32_t> m_shapes_;
    std::vector<uint32_t> m_strides_;
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

    void copy_from(const void* src, uint32_t size);

    Tensor reshape(std::vector<uint32_t> dims);

    void Reshape(std::vector<uint32_t> dims);

    uint32_t byte_size();
};
} // namespace core
} // namespace inferx
#endif