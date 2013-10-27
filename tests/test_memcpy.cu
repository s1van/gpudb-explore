// memory copy
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "test.h"
#include "gmm.h"

int test_memcpy()
{
	void *dptr, *dptr2, *ptr, *ptr2;
	size_t size = 1024 * 1024 * 10, i;
	int ret = 0;

	// Mallocs
	ptr = malloc(size);
	if (!ptr) {
		GMM_TPRINT("malloc failed for ptr\n");
		return -1;
	}

	ptr2 = malloc(size);
	if (!ptr2) {
		GMM_TPRINT("malloc failed for ptr2\n");
		free(ptr);
		return -1;
	}

	if (cudaMalloc(&dptr, size) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc for dptr failed\n");
		free(ptr2);
		free(ptr);
		return -1;
	}

	if (cudaMalloc(&dptr2, size) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc for dptr2 failed\n");
		cudaFree(dptr);
		free(ptr2);
		free(ptr);
		return -1;
	}

	for(i = 0; i < size; i += 4096) {
		*((char *)ptr + i) = 'x';
		*((char *)ptr2 + i) = 'y';
	}

	if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy HtoD failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("cudaMemcpyHostToDevice succeeded\n");

	if (cudaMemcpy(dptr2, dptr, size, cudaMemcpyDeviceToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy DtoD failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("cudaMemcpyDeviceToDevice succeeded\n");

	if (cudaMemcpy(ptr2, dptr2, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy DtoH failed\n");
		ret = -1;
		goto finish;
	}
	GMM_TPRINT("cudaMemcpyDeviceToHost succeeded\n");

	for(i = 0; i < size; i += 4096)
		if (*((char *)ptr2 + i) != 'x') {
			GMM_TPRINT("verification failed at i = %lu (*ptr2 = %c)\n", i, \
					*((char *)ptr2 + i));
			ret = -1;
			goto finish;
		}

	GMM_TPRINT("verification passed\n");

finish:
	if (cudaFree(dptr) != cudaSuccess) {
		GMM_TPRINT("cudaFree for dptr failed\n");
	}
	if (cudaFree(dptr2) != cudaSuccess) {
		GMM_TPRINT("cudaFree for dptr2 failed\n");
	}
	free(ptr);
	free(ptr2);

	return ret;
}
