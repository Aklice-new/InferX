#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

//_________________________CHECK ERROR_________________________//

// CUDA ERROR CHECK

void cuda_check(cudaError_t error, const char* file, const int line)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%i: %s\n", file, line, cudaGetErrorString(error));
        exit(-1);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// CUBLAS ERROR CHECK

void cublas_check(cublasStatus_t status, const char* file, const int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS error at %s:%i: %i\n", file, line, status);
        exit(-1);
    }
}
#define cublasCheck(err) (cublas_check(err, __FILE__, __LINE__))

#define CEIL_DIV(a, b) (((a) + (b) -1) / (b))
