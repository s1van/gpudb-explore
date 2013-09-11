#ifndef _GMM_H_
#define _GMM_H_

#include <driver_types.h>
#include <cuda_runtime_api.h>

// The GMM extensions to CUDA runtime interfaces, defined in gmm_interfaces.c
cudaError_t cudaReference(int which_arg);

#endif