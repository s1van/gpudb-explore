#include <stdlib.h>
#include <errno.h>

#include "common.h"
#include "client.h"
#include "core.h"
#include "util.h"


// CUDA function handlers, defined in gmm_interfaces.c
extern cudaError_t (*nv_cudaMalloc)(void **, size_t);
extern cudaError_t (*nv_cudaFree)(void *);
extern cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t,
		enum cudaMemcpyKind);
extern cudaError_t (*nv_cudaMemcpyAsync)(void *, const void *,
		size_t, enum cudaMemcpyKind, cudaStream_t stream);
extern cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *);
extern cudaError_t (*nv_cudaStreamSynchronize)(cudaStream_t);
extern cudaError_t (*nv_cudaSetupArgument) (const void *, size_t, size_t);
extern cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
extern cudaError_t (*nv_cudaMemset)(void * , int , size_t );
extern cudaError_t (*nv_cudaMemsetAsync)(void * , int , size_t, cudaStream_t);
extern cudaError_t (*nv_cudaDeviceSynchronize)(void);
extern cudaError_t (*nv_cudaLaunch)(void *);


struct gmm_local *plocal = NULL;


int gmm_local_init()
{
	if (plocal != NULL) {
		GMM_DPRINT("plocal already exists!\n");
		return -1;
	}

	plocal = (struct gmm_local *)malloc(sizeof(*plocal));
	if (!plocal) {
		GMM_DPRINT("failed to malloc for plocal: %s\n", strerror(errno));
		return -1;
	}

	initlock(&plocal->lock);
	plocal->size_alloced = 0;
	plocal->size_attached = 0;
	INIT_LIST_HEAD(&plocal->mems_alloced);
	INIT_LIST_HEAD(&plocal->mems_attached);

	return 0;
}

void gmm_local_fini()
{
	// TODO: have to free all memory objects still attached to device

	free(plocal);
	plocal = NULL;
}

// Allocate a new device memory object.
// We only allocate the host swap buffer space for now, and return the address
// of the host buffer to the user as the identifier of the object.
cudaError_t gmm_cudaMalloc(void **devPtr, size_t size)
{
	struct memobj *mem;

	if (size > devmem_size()) {
		GMM_DPRINT("request cudaMalloc size (%u) too large (dev: %ld)", \
				size, devmem_size());
		return cudaErrorMemoryAllocation;
	}

	mem = (struct memobj *)malloc(sizeof(*mem));
	if (!mem) {
		GMM_DPRINT("failed to malloc for memobj: %s\n", strerror(errno));
		return cudaErrorMemoryAllocation;
	}

	mem->addr_swap = malloc(size);
	if (!mem->addr_swap) {
		GMM_DPRINT("failed to malloc for host swap buffer: %s\n", \
				stderror(errno));
		free(mem);
		return cudaErrorMemoryAllocation;
	}

	mem->size = (long)size;
	mem->state = STATE_DETACHED;
	mem->addr_dev = NULL;
	mem->pinned = 0;
	list_alloced_add(plocal, mem);

	*devPtr = mem->addr_swap;
	return cudaSuccess;
}

cudaError_t gmm_cudaFree(void *devPtr)
{

}

cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{

}

cudaError_t gmm_cudaMemcpy(
		void *dst,
		const void *src,
		size_t count,
		enum cudaMemcpyKind kind)
{

}

cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{

}

cudaError_t gmm_cudaMemset(void * devPtr, int value, size_t count)
{

}

cudaError_t gmm_cudaLaunch(void* entry)
{

}
