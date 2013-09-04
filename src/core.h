#ifndef _GMM_CORE_H_
#define _GMM_CORE_H_

#include <stdint.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "list.h"
#include "spinlock.h"


// The local management info within each GMM client
struct gmm_local {
	struct spinlock lock;				// mutex for local synchronizations
	long size_alloced;					// total size of allocated mem
	long size_attached;					// total size of attached mem
	struct list_head list_alloced;		// list of all mem objects
	struct list_head list_attached;		// LRU list of attached mem objects
};

// State of a device memory object
typedef enum mem_state {
	STATE_ATTACHED = 0,		// object allocated with device memory
	STATE_DETACHED,			// object not attached with device memory
	STATE_FREEING,			// object being freed
	STATE_EVICTING,			// object being evicted
	STATE_EVICTED,
} mem_state_t;

// Device memory object
struct memobj {
	long size;				// size of the object in bytes
	mem_state_t state;		// state of the object
	void *addr_dev;			// device memory address
	void *addr_swap;		// host swap buffer address
	int pinned;				// atomic pin counter

	struct list_head entry_alloced;		// linked to the list of allocated
	struct list_head entry_attached;	// linked to the list of attached
};


// Functions exposed by GMM core
int gmm_local_init();
void gmm_local_fini();

cudaError_t gmm_cudaMalloc(void **devPtr, size_t size);
cudaError_t gmm_cudaFree(void *devPtr);
cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset);
cudaError_t gmm_cudaMemcpy(
		void *dst,
		const void *src,
		size_t count,
		enum cudaMemcpyKind kind);
cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream);
cudaError_t gmm_cudaMemset(void * devPtr, int value, size_t count);
cudaError_t gmm_cudaLaunch(void* entry);

#endif
