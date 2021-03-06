// local evictions
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "test.h"
#include "gmm.h"
#include "debug.h"

__global__ void kernel_inc(int *data, int count)
{
	int tot_threads = gridDim.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < count; i += tot_threads)
		data[i]++;
}

int test_evict_local()
{
	int *dptr = NULL, *dptr2 = NULL, *ptr = NULL;
	size_t size, sfree, total;
	int i, count, ret = 0;

	if (cudaMemGetInfo(&sfree, &total) != cudaSuccess) {
		GMM_TPRINT("failed to get mem info\n");
		return -1;
	}
	size = total * 3 / 4;
	count = size / sizeof(int);

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
	gmm_print_dptr(dptr);

	if (cudaMalloc(&dptr2, size) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc failed\n");
		cudaFree(dptr);
		free(ptr);
		return -1;
	}

	if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpyHostToDevice to dptr failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("cudaMemcpyHostToDevice succeeded\n");
	if (cudaMemcpy(dptr2, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpyHostToDevice to deptr2 failed\n");
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
	else
		GMM_TPRINT("1st kernel finished\n");

	gmm_print_dptr(dptr);

	if (cudaReference(0, HINT_DEFAULT) != cudaSuccess) {
		GMM_TPRINT("cudaReference failed\n");
		ret = -1;
		goto finish;
	}
	kernel_inc<<<256, 128>>>(dptr2, count);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		GMM_TPRINT("cudaThreadSynchronize returned error\n");
		ret = -1;
		goto finish;
	}
	else
		GMM_TPRINT("2nd kernel finished\n");

	gmm_print_dptr(dptr);

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

	if (cudaMemcpy(ptr, dptr2, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
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
	if (cudaFree(dptr2) != cudaSuccess) {
		GMM_TPRINT("cudaFree failed\n");
	}
	if (cudaFree(dptr) != cudaSuccess) {
		GMM_TPRINT("cudaFree failed\n");
	}
	free(ptr);

	return ret;
}
