#ifndef _GMM_H_
#define _GMM_H_

#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "hint.h"

// The GMM extensions to CUDA runtime interfaces. Interface
// implementations reside in interfaces.c.
cudaError_t cudaMallocEx(void **devPtr, size_t size, int flags);
cudaError_t cudaSetKernelPrio(int prio);
cudaError_t cudaReference(int which_arg);

#endif
