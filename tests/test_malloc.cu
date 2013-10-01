// memory region allocation
#include <stdio.h>
#include <cuda.h>

#include "test.h"
#include "gmm.h"

int test_malloc()
{
	void *dptr = NULL;
	size_t size = 1024;

	while (size < 1024L * 1024L * 513) {
		GMM_TPRINT("allocating %lu bytes\n", size);
		if (cudaMalloc(&dptr, size) != cudaSuccess) {
			GMM_TPRINT("cudaMalloc failed\n");
			return -1;
		}
		GMM_TPRINT("dptr = %p\n", dptr);
		if (cudaFree(dptr) != cudaSuccess) {
			GMM_TPRINT("cudaFree failed\n");
			return -1;
		}
		GMM_TPRINT("region freed\n");
		size *= 2;
	}

	return 0;
}
