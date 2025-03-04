/**
 * @file clamp_cuda.cu
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "core/cuda_helper.h"
#include "layer/kernels/clamp.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;

__global__ void clamp_kernel(const float* input_ptr, float* output_ptr, int N, float min_val, float max_val)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }
    output_ptr[idx] = max(min_val, min(max_val, input_ptr[idx]));
}

StatusCode ClampLayer::forward_gpu()
{
    const float min_val = min_val_;
    const float max_val = max_val_;
    const uint32_t size = inputs_[0]->size();
    auto input = inputs_[0];
    auto output = outputs_[0];

    input->to_cuda();
    output->to_cuda();

    dim3 thread_per_block = 256;
    dim3 block_per_grid = (size + 255) / 256;
    clamp_kernel<<<block_per_grid, thread_per_block>>>(
        input->ptr<float>(), output->ptr<float>(), size, min_val, max_val);
    cudaCheck(cudaGetLastError());

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx