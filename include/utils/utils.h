#ifndef _UTILS_H_
#define _UTILS_H_

#include <chrono>
#include <omp.h>
#include <string>
#include <glog/logging.h>

namespace inferx
{

#define INFER_LOG(format, ...) fprintf(stderr, format, ##__VA_ARGS__)
#define INFER_ERROR(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

#define CUDA_CHECK(expr)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (expr);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            auto error = cudaGetErrorString(err);                                                                      \
            INFER_ERROR(                                                                                               \
                "CUDA error %d: %s, at file : %s \n"                                                                   \
                "line %d : %s ",                                                                                       \
                err, error, __FILE__, __LINE__, __PRETTY_FUNCTION__);                                                  \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(err)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t err_ = (err);                                                                                   \
        if (err_ != CUBLAS_STATUS_SUCCESS)                                                                             \
        {                                                                                                              \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);                                 \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

static const int MAX_THREADS = omp_get_max_threads();

void read_data_from_txt(const std::string& file_path, float* data, size_t size);

class Timer
{
    using Time = std::chrono::high_resolution_clock;
    using ms = std::chrono::milliseconds;
    using fsec = std::chrono::duration<float>;

public:
    Timer()
    {
        start = Time::now();
    }
    double get_time()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<fsec>(end - start);
        return duration.count();
    };

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
} // namespace inferx

#endif