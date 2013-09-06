#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>

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


struct gmm_context *pcontext = NULL;


int gmm_context_init()
{
	if (pcontext != NULL) {
		GMM_DPRINT("pcontext already exists!\n");
		return -1;
	}

	pcontext = (struct gmm_context *)malloc(sizeof(*pcontext));
	if (!pcontext) {
		GMM_DPRINT("failed to malloc for pcontext: %s\n", strerror(errno));
		return -1;
	}

	initlock(&pcontext->lock);
	pcontext->size_alloced = 0;
	pcontext->size_attached = 0;
	INIT_LIST_HEAD(&pcontext->list_alloced);
	INIT_LIST_HEAD(&pcontext->list_attached);

	return 0;
}

void gmm_context_fini()
{
	// TODO: have to free all memory objects still attached to device

	free(pcontext);
	pcontext = NULL;
}

// Allocate a new device memory object.
// We only allocate the host swap buffer space for now, and return the address
// of the host buffer to the user as the identifier of the object.
// TODO: flags can pass whether object should be always pinned and the static
// read-write hints.
cudaError_t gmm_cudaMalloc(void **devPtr, size_t size, int flags)
{
	struct region *mem;
	int nblocks;

	if (size > dev_memsize()) {
		GMM_DPRINT("request cudaMalloc size (%u) too large (dev: %ld)", \
				size, devmem_size());
		return cudaErrorMemoryAllocation;
	}

	mem = (struct region *)malloc(sizeof(*mem));
	if (!mem) {
		GMM_DPRINT("failed to malloc for memobj: %s\n", strerror(errno));
		return cudaErrorMemoryAllocation;
	}

	mem->addr_swp = mmap(NULL, size, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
	if (!mem->addr_swp) {
		GMM_DPRINT("failed to malloc for host swap buffer: %s\n", \
				stderror(errno));
		free(mem);
		return cudaErrorMemoryAllocation;
	}

	nblocks = NRBLOCKS(size);
	mem->blocks = (struct block *)malloc(sizeof(struct block) * nblocks);
	if (!mem->blocks) {
		GMM_DPRINT("failed to malloc for blocks array: %s\n", strerror(errno));
		munmap(mem->addr_swp, size);
		free(mem);
		return cudaErrorMemoryAllocation;
	}
	memset(mem->blocks, 0, sizeof(struct block) * nblocks);

	mem->size = (long)size;
	mem->addr_dev = NULL;
	initlock(&mem->lock);
	mem->state = STATE_DETACHED;
	atomic_set(&mem->pinned, 0);

	list_alloced_add(pcontext, mem);
	*devPtr = mem->addr_swp;

	return cudaSuccess;
}

int gmm_free(struct memobj *m);

cudaError_t gmm_cudaFree(void *devPtr)
{
	struct region *rgn;

	if (!(rgn = region_lookup(pcontext, devPtr))) {
		GDEV_DPRINT("cannot find memory object with devPtr %p\n", devPtr);
		return cudaErrorInvalidDevicePointer;
	}

	if (gmm_free(rgn) < 0)
		return cudaErrorUnknown;
	else
		return cudaSuccess;
}

int gmm_htod_attached(
		struct region *r,
		void *dst,
		const void *src,
		size_t count);

cudaError_t gmm_cudaMemcpyHtoD(
		void *dst,
		const void *src,
		size_t count)
{
	struct region *r;

	if (count <= 0)
		return cudaErrorInvalidValue;

	r = region_lookup(pcontext, dst);
	if (!r) {
		GMM_DPRINT("could not find device memory region containing %p\n", dst);
		return cudaErrorInvalidDevicePointer;
	}
	if (r->state == STATE_FREEING) {
		GMM_DPRINT("region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (dst + count > r->addr_swp + r->size) {
		GMM_DPRINT("copy overflow\n");
		return cudaErrorInvalidValue;
	}

	// Copy/COW data to the region's host swap buffer
	if (r->state == STATE_DETACHED) {
		int iblock = BLOCKIDX(dst - r->addr_swp);
		int iend = BLOCKIDX(dst + (count - 1) - r->addr_swp);

		memcpy(dst, src, count);
		while (iblock <= iend) {
			r->blocks[iblock].swp_valid = 1;
			// xxx.dev_valid must be 0
			iblock++;
		}
	}
	else {
		if (gmm_htod_attached(r, dst, src, count) < 0)
			return cudaErrorUnknown;
	}

	return cudaSuccess;
}

cudaError_t gmm_cudaMemcpyDtoH(
		void *dst,
		const void *src,
		size_t count)
{

}

cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{

}

cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{

}

cudaError_t gmm_cudaLaunch(void* entry)
{

}

cudaError_t gmm_cudaMemset(void * devPtr, int value, size_t count)
{

}

// TODO
static int gmm_free(struct region *r)
{
	int being_evicted = 0;

re_acquire:
	acquire(&r->lock);

	switch (r->state) {
	case STATE_ATTACHED:
		break;
	case STATE_EVICTING:
		break;
	case STATE_FREEING:
		break;
	default:
		break;
	}
	release(&r->lock);

	if (being_evicted) {
#ifdef GMM_CONFIG_FREE_OPTIMIZE
		return 0;
#else
		being_evicted = 0;
		goto re_acquire;
#endif
	}

	// Now this memory region can be freed
	list_alloced_del(r);
	free(r->blocks);
	munmap(r->addr_swp, r->size);
	if (r->state == STATE_ATTACHED && r->addr_dev) {
		nv_cudaFree(r->addr_dev);
		// TODO: update local and global info
	}
	free(r);

	return 1;
}

// Sync the host and device data copies of a block.
// The direction of sync is determined by current valid flags. Data are synced
// from the valid copy to the invalid copy.
static void gmm_block_sync(struct region *r, int block)
{
	int dvalid = r->blocks[block].dev_valid;
	int svalid = r->blocks[block].swp_valid;
	unsigned long off1, off2;

	// Nothing to sync if both are valid or invalid
	if (dvalid ^ svalid == 0)
		return;

	off1 = block * BLOCKSIZE;
	off2 = MIN(off1 + BLOCKSIZE, r->size);
	if (dvalid && !svalid) {
		// Sync from device to host swap buffer
		gmm_memcpy_dtoh(r->addr_swp + off1, r->addr_dev + off1, off2 - off1);
		r->blocks[block].swp_valid = 1;
	}
	else {
		// Sync from host swap buffer to device
		gmm_memcpy_htod(r->addr_dev + off1, r->addr_swp + off1, off2 - off1);
		r->blocks[block].dev_valid = 1;
	}
}

// Copy a piece of data from $src to (the host swap buffer of) a block in $r.
// $offset gives the offset of the destination address relative to $r->addr_swp.
// $size is the size of data to be copied.
// $block tells which block is being modified.
// $skip specifies whether to skip copying if the block is being locked.
// $skipped, if not null, returns whether skipped (1 - skipped; 0 - not skipped)
static void gmm_htod_block(
		struct region *r,
		unsigned long offset,
		void *src,
		unsigned long size,
		int block,
		int skip,
		char *skipped)
{
	// partial over-writing?
	if ((offset % BLOCKSIZE) || (size < BLOCKSIZE && offset + size < r->size)) {
		if (r->blocks[block].swp_valid || !r->blocks[block].dev_valid) {
			// no locking needed
			memcpy(r->addr_swp + offset, src, size);
			if (!r->blocks[block].swp_valid)
				r->blocks[block].swp_valid = 1;
			if (r->blocks[block].dev_valid)
				r->blocks[block].dev_valid = 0;
		}
		else {
			// locking needed
			while (!try_acquire(&r->blocks[block].lock)) {
				if (skip) {
					if (skipped)
						*skipped = 1;
					return;
				}
			}
			if (r->blocks[block].swp_valid || !r->blocks[block].dev_valid) {
				memcpy(r->addr_swp + offset, src, size);
				if (!r->blocks[block].swp_valid)
					r->blocks[block].swp_valid = 1;
				if (r->blocks[block].dev_valid)
					r->blocks[block].dev_valid = 0;
				release(&r->blocks[block].lock);
			}
			else {
				gmm_block_sync(r, block);
				release(&r->blocks[block].lock);
				memcpy(r->addr_swp + offset, src, size);
				r->blocks[block].dev_valid = 0;
			}
			if (skipped)
				*skipped = 0;
		}
	}
	else {
	}
}

// Handle HtoD data transfer to a region that is attached with device memory.
// Note: the region is state may enter/leave STATE_EVICTING any time during the
// the handling process.
//
// Over-writing a whole block is different from modifying a block partially.
// The former can be handled by invalidating the dev copy of the block
// and setting the swp copy valid; the later requires a sync of the dev
// copy to the swp, if the dev has a newer copy, before the data can be
// written to the swp.
//
// Here we provide two implementations:
// (1) a ``transactional'' solution, which means, if anything goes wrong during
// the copying, the states of the rest blocks are still correct. The drawback
// of this implementation is that it is conservative and may not fully exploit
// available concurrency in the system.
// (2) a radical solution that fully exploits possible concurrency.
//
// Following is the radical solution.
static int gmm_htod_attached(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	int iblock, ifirst, ilast;
	unsigned long offset;
	void *s = src;
	char *skipped;

	offset = (unsigned long)(dst - r->addr_swp);
	ifirst = BLOCKIDX(offset);
	ilast = BLOCKIDX(offset + (count - 1));
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		GMM_DPRINT("failed to malloc for copied[]: %s\n", strerrno(errno));
		return -1;
	}

	// For each full-block over-writing, set dev_valid=0, swp_valid=1.
	// Since we know the memory range being over-written, setting flags ahead
	// help prevent the evicter from wasting time evicting those blocks. This
	// is one unique advantage of us compared with CPU memory management.
	if (ifirst == ilast && count == BLOCKSIZE) {
		r->blocks[ifirst].dev_valid = 0;
		r->blocks[ifirst].swp_valid = 1;
	}
	else if (ifirst < ilast) {
		if (offset % BLOCKSIZE == 0) {
			r->blocks[ifirst].dev_valid = 0;
			r->blocks[ifirst].swp_valid = 1;
		}
		if ((offset + count) % BLOCKSIZE == 0 || (offset + count == r->size)) {
			r->blocks[ilast].dev_valid = 0;
			r->blocks[ilast].swp_valid = 1;
		}
		for (iblock = ifirst + 1; iblock < ilast; iblock++) {
			r->blocks[iblock].dev_valid = 0;
			r->blocks[iblock].swp_valid = 1;
		}
	}

	// Then, copy data block by block, skipping blocks that are locked (very
	// likely being evicted). skipped[] records whether each block was actually
	// copied.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long end = BLOCKUP(offset);
		end = MIN(end, (unsigned long)(dst - r->addr_swp + count));
		gmm_htod_block(r, offset, s, end - offset, iblock, 1,
				skipped + (iblock - ifirst));
		s += end - offset;
		offset = end;
	}

	// Finally, copy the rest blocks, blocking if an uncopied block is locked
	offset = (unsigned long)(dst - r->addr_swp);
	s = src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long end = BLOCKUP(offset);
		end = MIN(end, (unsigned long)(dst - r->addr_swp + count));
		if (skipped[iblock - ifirst])
			gmm_htod_block(r, offset, s, end - offset, iblock, 0, NULL);
		s += end - offset;
		offset = end;
	}

	free(skipped);
	return 0;
}
