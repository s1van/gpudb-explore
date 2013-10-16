// kernel launch
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "test.h"
#include "gmm.h"

__global__ void kernel_inc(int *data, int count)
{
    int tot_threads = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < count; i += tot_threads)
    	data[i] = 1;
}

int test_launch()
{
	int *dptr = NULL, *ptr = NULL;
	int count = 1024 * 1024 * 10;
	size_t size = sizeof(int) * count;
	int i, ret = 0;

	ptr = (int *)malloc(size);
	if (!ptr) {
		GMM_TPRINT("malloc failed for ptr\n");
		return -1;
	}
	memset(ptr, 0, size);

	if (cudaMalloc(&dptr, size) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc failed\n");
		free(ptr);
		return -1;
	}

	if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpyHostToDevice failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("cudaMemcpyHostToDevice succeeded\n");

	if (cudaReference(0, HINT_DEFAULT) != cudaSuccess) {
		GMM_TPRINT("cudaReference failed\n");
		ret = -1;
		goto finish;
	}

	kernel_inc<<<256, 128>>>(dptr, count);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		GMM_TPRINT("cudaThreadSynchronize returned error\n");
		ret = -1;
		goto finish;
	}

	if (cudaMemcpy(ptr, dptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy DtoH failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("cudaMemcpyDeviceToHost succeeded\n");

	for(i = 0; i < count; i++)
		if (ptr[i] != 1) {
			GMM_TPRINT("verification failed at ptr[%d]==%d\n", i, ptr[i]);
			ret = -1;
			goto finish;
		}
	GMM_TPRINT("verification passed\n");

finish:
	if (cudaFree(dptr) != cudaSuccess) {
		GMM_TPRINT("cudaFree failed\n");
	}
	free(ptr);

	return ret;
}
