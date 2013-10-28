// dptr array regions
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "test.h"
#include "gmm.h"

__global__ void kernel_inc(int **data, int dim, int count)
{
	int tot_threads = gridDim.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < count; i += tot_threads) {
		for (int j = 0; j < dim; j++)
			data[j][i]++;
	}
}

int test_ptarray()
{
	int **dptr = NULL, *dptr1 = NULL, *dptr2 = NULL, *ptr = NULL;
	int count = 1000 * 1000 * 10;
	size_t size = sizeof(int) * count;
	int i, ret = 0;

	ptr = (int *)malloc(size);
	if (!ptr) {
		GMM_TPRINT("malloc failed for ptr\n");
		return -1;
	}
	memset(ptr, 0, size);

	if (cudaMalloc(&dptr1, size) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc failed\n");
		free(ptr);
		return -1;
	}
	if (cudaMalloc(&dptr2, size) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc failed\n");
		cudaFree(dptr1);
		free(ptr);
		return -1;
	}
	if (cudaMallocEx((void **)&dptr, sizeof(int *) * 2, HINT_PTARRAY) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc failed\n");
		cudaFree(dptr2);
		cudaFree(dptr1);
		free(ptr);
		return -1;
	}

	if (cudaMemcpy(dptr1, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpyHostToDevice failed\n");
		ret = -1;
		goto finish;
	}
	if (cudaMemcpy(dptr2, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpyHostToDevice failed\n");
		ret = -1;
		goto finish;
	}
	if (cudaMemcpy(dptr, &dptr1, sizeof(int *), cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpyHostToDevice failed\n");
		ret = -1;
		goto finish;
	}
	if (cudaMemcpy(dptr + 1, &dptr2, sizeof(int *), cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpyHostToDevice failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("cudaMemcpyHostToDevice succeeded\n");

	i = 1;
	do {
		if (cudaReference(0, HINT_READ | HINT_PTARRAY | HINT_PTADEFAULT) != cudaSuccess) {
			GMM_TPRINT("cudaReference failed\n");
			ret = -1;
			goto finish;
		}
		kernel_inc<<<256, 128>>>(dptr, 2, count);
	} while (i-- > 0);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		GMM_TPRINT("cudaThreadSynchronize returned error\n");
		ret = -1;
		goto finish;
	}
	else
		GMM_TPRINT("kernel finished\n");

	if (cudaMemcpy(ptr, dptr1, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy DtoH failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("first cudaMemcpyDeviceToHost succeeded\n");

	for(i = 0; i < count; i++)
		if (ptr[i] != 2) {
			GMM_TPRINT("verification failed at ptr[%d]==%d\n", i, ptr[i]);
			ret = -1;
			goto finish;
		}

	if (cudaMemcpy(ptr, dptr2, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy DtoH failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("second cudaMemcpyDeviceToHost succeeded\n");

	for(i = 0; i < count; i++)
		if (ptr[i] != 2) {
			GMM_TPRINT("verification failed at ptr[%d]==%d\n", i, ptr[i]);
			ret = -1;
			goto finish;
		}

	GMM_TPRINT("verification passed\n");

finish:
	if (cudaFree(dptr) != cudaSuccess) {
		GMM_TPRINT("cudaFree failed\n");
	}
	if (cudaFree(dptr2) != cudaSuccess) {
		GMM_TPRINT("cudaFree failed\n");
	}
	if (cudaFree(dptr1) != cudaSuccess) {
		GMM_TPRINT("cudaFree failed\n");
	}
	free(ptr);

	return ret;
}
