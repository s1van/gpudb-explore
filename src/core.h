#ifndef _GMM_CORE_H_
#define _GMM_CORE_H_

#include <stdint.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "list.h"
#include "spinlock.h"
#include "atomic.h"


// The local management info within each GMM client
struct gmm_context {
	struct spinlock lock;				// mutex for local synchronizations
	long size_alloced;					// total size of allocated mem
	long size_attached;					// total size of attached mem
	struct list_head list_alloced;		// list of all mem objects
	struct list_head list_attached;		// LRU list of attached mem objects
};

// State of a device memory region
typedef enum region_state {
	STATE_ATTACHED = 0,		// object allocated with device memory
	STATE_DETACHED,			// object not attached with device memory
	STATE_FREEING,			// object being freed
	STATE_EVICTING,			// object being evicted
	STATE_EVICTED
} region_state_t;

// Device memory block.
#define BLOCKSIZE		(4096 * 1024)
struct block {
	int dev_valid;			// if data copy on device is valid
	int swp_valid;			// if data copy in host swap buffer is valid
	struct spinlock lock;	// r/w lock
};

// Device memory region.
// A device memory region is a virtual memory area allocated by the user
// program through cudaMalloc. It is logically partitioned into an array of
// fixed-length device memory blocks. Due to lack of system support, all blocks
// must be attached/detached together. But the valid and dirty statuses of
// each block are maintained separately.
struct region {
	long size;				// size of the object in bytes
	void *addr_dev;			// device memory address
	void *addr_swp;			// host swap buffer address
	struct block *blocks;	// device memory blocks
	struct spinlock lock;	// the lock that protects memory object state
	region_state_t state;	// state of the object
	atomic_t pinned;		// atomic pin counter

	struct list_head entry_alloced;		// linked to the list of allocated
	struct list_head entry_attached;	// linked to the list of attached
};


#define NRBLOCKS(size)		(((size) + BLOCKSIZE - 1) / BLOCKSIZE)
#define BLOCKIDX(offset)	((unsigned long)(offset) / BLOCKSIZE)
#define BLOCKUP(offset)		((offset + BLOCKSIZE) / BLOCKSIZE * BLOCKSIZE)

// Functions exposed by GMM core
int gmm_context_init();
void gmm_context_fini();

cudaError_t gmm_cudaMalloc(void **devPtr, size_t size, int flags);
cudaError_t gmm_cudaFree(void *devPtr);
cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset);
cudaError_t gmm_cudaMemcpyHtoD(
		void *dst,
		const void *src,
		size_t count);
cudaError_t gmm_cudaMemcpyDtoH(
		void *dst,
		const void *src,
		size_t count);
cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream);
cudaError_t gmm_cudaMemset(void * devPtr, int value, size_t count);
cudaError_t gmm_cudaLaunch(void* entry);

#endif
