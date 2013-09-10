#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sched.h>

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

static int gmm_free(struct memobj *m);
static int gmm_htod(
		struct region *r,
		void *dst,
		const void *src,
		size_t count);
static int gmm_dtoh(
		struct region *r,
		void *dst,
		const void *src,
		size_t count);


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
	//pcontext->size_alloced = 0;
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

	if (size > get_memsize()) {
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

cudaError_t gmm_cudaFree(void *devPtr)
{
	struct region *r;

	if (!(r = region_lookup(pcontext, devPtr))) {
		GDEV_DPRINT("cannot find memory object with devPtr %p\n", devPtr);
		return cudaErrorInvalidDevicePointer;
	}

	if (gmm_free(r) < 0)
		return cudaErrorUnknown;
	else
		return cudaSuccess;
}

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

	if (gmm_htod(r, dst, src, count) < 0)
		return cudaErrorUnknown;

	return cudaSuccess;
}

cudaError_t gmm_cudaMemcpyDtoH(
		void *dst,
		const void *src,
		size_t count)
{
	struct region *r;

	if (count <= 0)
		return cudaErrorInvalidValue;

	r = region_lookup(pcontext, src);
	if (!r) {
		GMM_DPRINT("could not find device memory region containing %p\n", src);
		return cudaErrorInvalidDevicePointer;
	}
	if (r->state == STATE_FREEING) {
		GMM_DPRINT("region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (src + count > r->addr_swp + r->size) {
		GMM_DPRINT("device memory access out of boundary\n");
		return cudaErrorInvalidValue;
	}

	if (gmm_dtoh(r, dst, src, count) < 0)
		return cudaErrorUnknown;

	return cudaSuccess;
}

// For passing reference hints before each kernel launch
int arguments[NARGUMENTS];
int nargs = 0;

// Pass reference hints. $which_arg tells which argument in the following
// cudaSetupArgument calls is a device memory address.
cudaError_t cudaReference(int which_arg)
{
	if (nargs < NARGUMENTS)
		arguments[nargs++] = which_arg;
	else {
		GMM_DPRINT("too many reference hints for a kernel (max %d)\n", \
				NARGUMENTS);
		return cudaErrorInvalidValue;
	}

	return cudaSuccess;
}

// TODO: either do something really useful or simply delete this function
cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{
	return nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

// CUDA pushes kernel arguments from left to right. For example, for a kernel
//		k(a, b, c)
// , a will be called with gmm_cudaSetupArgument first, followed by b and
// finally c. $offset gives the actual offset of an argument in the call stack,
// rather than which argument being pushed.
//
// Use reference hints if any. Otherwise, parse automatically (but there may be
// parsing errors).
cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{

}

cudaError_t gmm_cudaLaunch(void* entry)
{
	// First handle detached regions

	// Then reset reference hints
	nargs = 0;

	// Finally launch kernel
}

cudaError_t gmm_cudaMemset(void * devPtr, int value, size_t count)
{

}

// The return value of this function tells whether the region has been
// immediately freed. 0 - not freed yet; 1 - freed.
static int gmm_free(struct region *r)
{
	int being_evicted = 0;

	// First, properly inspect/set region state
re_acquire:
	acquire(&r->lock);
	switch (r->state) {
	case STATE_ATTACHED:
		if (!region_pinned(r))
			list_attached_del(pcontext, r);
		else {
			release(&r->lock);
			sched_yield();
			goto re_acquire;
		}
		break;
	case STATE_EVICTING:
		// Tell the evictor that this region is being freed
		r->state = STATE_FREEING;
		release(&r->lock);
		return 0;
		break;
	case STATE_FREEING:
		// The evictor has not seen this region being freed
		release(&r->lock);
		sched_yield();
		goto re_acquire;
		break;
	case STATE_DETACHED:
		break;
	default:
		panic("unknown region state (in gmm_free)");
		break;
	}
	release(&r->lock);

	// Now, this memory region can be freed
	list_alloced_del(pcontext, r);
	if (r->blocks)
		free(r->blocks);
	if (r->addr_swp)
		munmap(r->addr_swp, r->size);
	if (r->state == STATE_ATTACHED && r->addr_dev) {
		nv_cudaFree(r->addr_dev);
		atomic_subl(&pcontext->size_attached, r->size);
		update_attached(-r->size);
		update_detachable(-r->size);
	}
	free(r);

	return 1;
}

// TODO
static void gmm_memcpy_dtoh(void *dst, void *src, unsigned long size)
{
	// Use nv_cudaMemcpyAsync
}

// TODO
static void gmm_memcpy_htod(void *dst, void *src, unsigned long size)
{
	// Use nv_cudaMemcpyAsync
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
#ifdef GMM_CONFIG_HTOD_RADICAL
// This is an implementation matching the radical version of gmm_htod.
static void gmm_htod_block(
		struct region *r,
		unsigned long offset,
		void *src,
		unsigned long size,
		int block,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + block;

	// partial modification
	if ((offset % BLOCKSIZE) || (size < BLOCKSIZE && offset + size < r->size)) {
		if (b->swp_valid || !b->dev_valid) {
			// no locking needed
			memcpy(r->addr_swp + offset, src, size);
			if (!b->swp_valid)
				b->swp_valid = 1;
			if (b->dev_valid)
				b->dev_valid = 0;
		}
		else {
			// locking needed
			while (!try_acquire(&b->lock)) {
				if (skip) {
					if (skipped)
						*skipped = 1;
					return;
				}
			}
			if (b->swp_valid || !b->dev_valid) {
				release(&r->blocks[block].lock);
				memcpy(r->addr_swp + offset, src, size);
				if (!b->swp_valid)
					b->swp_valid = 1;
				if (b->dev_valid)
					b->dev_valid = 0;
			}
			else {
				// We don't need to pin the device memory because we are
				// holding the lock of a swp_valid=0,dev_valid=1 block, which
				// will prevent the evictor, if any, from freeing the device
				// memory under us.
				gmm_block_sync(r, block);
				release(&b->lock);
				memcpy(r->addr_swp + offset, src, size);
				b->dev_valid = 0;
			}
		}
	}
	// full over-writing (its valid flags have been set in advance)
	else {
		while (!try_acquire(&b->lock)) {
			if (skip) {
				if (skipped)
					*skipped = 1;
				return;
			}
		}
		// acquire the lock and release immediately, to avoid data races with
		// the evictor who's writing swp buffer
		release(&r->blocks[block].lock);
		memcpy(r->addr_swp + offset, src, size);
	}

	if (skipped)
		*skipped = 0;
}
#else
// This is an implementation matching the conservative version of gmm_htod.
static void gmm_htod_block(
		struct region *r,
		unsigned long offset,
		void *src,
		unsigned long size,
		int block,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + block;
	int partial = (offset % BLOCKSIZE) ||
			(size < BLOCKSIZE && offset + size < r->size);

	if (b->swp_valid || !b->dev_valid) {
		// no locking needed
		memcpy(r->addr_swp + offset, src, size);
		if (!b->swp_valid)
			b->swp_valid = 1;
		if (b->dev_valid)
			b->dev_valid = 0;
	}
	else {
		// locking needed
		while (!try_acquire(&b->lock)) {
			if (skip) {
				if (skipped)
					*skipped = 1;
				return;
			}
		}
		if (b->swp_valid || !b->dev_valid) {
			release(&r->blocks[block].lock);
			memcpy(r->addr_swp + offset, src, size);
			if (!b->swp_valid)
				b->swp_valid = 1;
			if (b->dev_valid)
				b->dev_valid = 0;
		}
		else {
			if (partial) {
				// We don't need to pin the device memory because we are
				// holding the lock of a swp_valid=0,dev_valid=1 block, which
				// will prevent the evictor, if any, from freeing the device
				// memory under us.
				gmm_block_sync(r, block);
				release(&b->lock);
			}
			else {
				b->swp_valid = 1;
				release(&b->lock);
			}
			memcpy(r->addr_swp + offset, src, size);
			b->dev_valid = 0;
		}
	}

	if (skipped)
		*skipped = 0;
}
#endif

// Handle a HtoD data transfer request.
// Note: the region may enter/leave STATE_EVICTING any time.
//
// Over-writing a whole block is different from modifying a block partially.
// The former can be handled by invalidating the dev copy of the block
// and setting the swp copy valid; the later requires a sync of the dev
// copy to the swp, if the dev has a newer copy, before the data can be
// written to the swp.
//
// Block-based memory management improves concurrency. Another important factor
// to consider is to reduce unnecessary swapping, if the region is being
// evicted during an HtoD action. Here we provide two implementations:
// one is radical, the other is conservative.
#if defined(GMM_CONFIG_HTOD_RADICAL)
// The radical version
static int gmm_htod(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	int iblock, ifirst, ilast;
	unsigned long off, end;
	void *s = src;
	char *skipped;

	off = (unsigned long)(dst - r->addr_swp);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(off + (count - 1));
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		GMM_DPRINT("failed to malloc for skipped[]: %s\n", strerrno(errno));
		return -1;
	}

	// For each full-block over-writing, set dev_valid=0 and swp_valid=1.
	// Since we know the memory range being over-written, setting flags ahead
	// help prevent the evictor, if there is one, from wasting time evicting
	// those blocks. This is one unique advantage of us compared with CPU
	// memory management, where the OS usually does not have such interfaces
	// or knowledge.
	if (ifirst == ilast && count == BLOCKSIZE) {
		r->blocks[ifirst].dev_valid = 0;
		r->blocks[ifirst].swp_valid = 1;
	}
	else if (ifirst < ilast) {
		if (off % BLOCKSIZE == 0) {
			r->blocks[ifirst].dev_valid = 0;
			r->blocks[ifirst].swp_valid = 1;
		}
		if (end % BLOCKSIZE == 0 || end == r->size) {
			r->blocks[ilast].dev_valid = 0;
			r->blocks[ilast].swp_valid = 1;
		}
		for (iblock = ifirst + 1; iblock < ilast; iblock++) {
			r->blocks[iblock].dev_valid = 0;
			r->blocks[iblock].swp_valid = 1;
		}
	}

	// Then, copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted). skipped[]
	// records whether a block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		gmm_htod_block(r, off, s, size, iblock, 1, skipped + (iblock - ifirst));
		s += size;
		off += size;
	}

	// Finally, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->addr_swp);
	s = src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst])
			gmm_htod_block(r, off, s, size, iblock, 0, NULL);
		s += size;
		off += size;
	}

	free(skipped);
	return 0;
}
#else
// The conservative version
static int gmm_htod(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	unsigned long off, end, size;
	int ifirst, ilast, iblock;
	char *skipped;
	void *s = src;

	off = (unsigned long)(dst - r->addr_swp);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		GMM_DPRINT("failed to malloc for skipped[]: %s\n", strerrno(errno));
		return -1;
	}

	// Copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted). skipped[]
	// records whether a block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		gmm_htod_block(r, off, s, size, iblock, 1, skipped + (iblock - ifirst));
		s += size;
		off += size;
	}

	// Then, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->addr_swp);
	s = src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst])
			gmm_htod_block(r, off, s, size, iblock, 0, NULL);
		s += size;
		off += size;
	}

	free(skipped);
	return 0;
}
#endif

static int gmm_dtoh_block(
		void *dst,
		struct region *r,
		unsigned long off,
		unsigned long size,
		int iblock,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + iblock;

	if (b->swp_valid) {
		memcpy(dst, r->addr_swp + off, size);
		return 0;
	}

	while (!try_acquire(&b->lock)) {
		if (skip) {
			if (skipped)
				*skipped = 1;
			return 0;
		}
	}

	if (b->swp_valid) {
		release(&b->lock);
		memcpy(dst, r->addr_swp + off, size);
	}
	else if (!b->dev_valid) {
		release(&b->lock);
	}
	else if (skip) {
		release(&b->lock);
		if (skipped)
			*skipped = 1;
		return 0;
	}
	else {
		// We don't need to pin the device memory because we are holding the
		// lock of a swp_valid=0,dev_valid=1 block, which will prevent the
		// evictor, if any, from freeing the device memory under us.
		gmm_block_sync(r, iblock);
		release(&b->lock);
		memcpy(dst, r->addr_swp + off, size);
	}

	if (skipped)
		*skipped = 0;

	return 0;
}

// TODO: It is possible to achieve pipelined copying, i.e., copy a block from
// its host swap buffer to user buffer while the next block is being fetched
// from device memory.
static int gmm_dtoh(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	unsigned long off = (unsigned long)(src - r->addr_swp);
	unsigned long end = off + count, size;
	int ifirst = BLOCKIDX(off), iblock;
	char *skipped;

	skipped = (char *)malloc(BLOCKIDX(end - 1) - ifirst + 1);
	if (!skipped) {
		GMM_DPRINT("failed to malloc for skipped[]: %s\n", strerrno(errno));
		return -1;
	}

	// First, copy blocks whose swp buffers contain immediate, valid data
	iblock = ifirst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		gmm_dtoh_block(dst, r, off, size, iblock, 1, skipped + iblock - ifirst);
		dst += size;
		off += size;
		iblock++;
	}

	// Then, copy the rest blocks
	off = (unsigned long)(src - r->addr_swp);
	iblock = ifirst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst])
			gmm_dtoh_block(dst, r, off, size, iblock, 0, NULL);
		dst += size;
		off += size;
		iblock++;
	}

	free(skipped);
	return 0;
}
