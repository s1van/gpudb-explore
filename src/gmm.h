#ifndef _GMM_H_
#define _GMM_H_

#include <driver_types.h>
#include <cuda_runtime_api.h>

#define HINT_READ		1
#define HINT_WRITE		2

// The GMM extensions to CUDA runtime interfaces
cudaError_t cudaReference(int which_arg);

#endif
