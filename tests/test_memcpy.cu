// memory copy
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "test.h"
#include "gmm.h"

int test_memcpy()
{
	void *dptr = NULL, *ptr, *ptr2;
	size_t size = 1024 * 1024 * 10;
	int i, ret = 0;

	ptr = malloc(size);
	if (!ptr) {
		GMM_TPRINT("malloc failed for ptr\n");
		return -1;
	}
	for(i = 0; i < size; i += 4096)
		*(char *)(ptr + i) = 'x';

	if (cudaMalloc(&dptr, size) != cudaSuccess) {
		GMM_TPRINT("cudaMalloc failed\n");
		free(ptr);
		return -1;
	}

	if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy HtoD failed\n");
		ret = -1;
		goto finish;
	}

	ptr2 = malloc(size);
	if (!ptr2) {
		GMM_TPRINT("malloc failed for ptr2\n");
		ret = -1;
		goto finish;
	}

	if (cudaMemcpy(ptr2, dptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		GMM_TPRINT("cudaMemcpy DtoH failed\n");
		free(ptr2);
		ret = -1;
		goto finish;
	}

	for(i = 0; i < size; i += 4096)
		if (*(char *)(ptr2 + i) != 'x') {
			GMM_TPRINT("verification failed at i = %d (*ptr2 = %d)\n", i, \
					*(char *)(ptr2 + i));
			free(ptr2);
			ret = -1;
			goto finish;
		}

	GMM_TPRINT("verification passed\n");
	free(ptr2);

finish:
	if (cudaFree(dptr) != cudaSuccess) {
		GMM_TPRINT("cudaFree failed\n");
	}
	free(ptr);

	return ret;
}
