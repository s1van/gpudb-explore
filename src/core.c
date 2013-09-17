#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sched.h>

#include "common.h"
#include "client.h"
#include "core.h"
#include "util.h"
#include "hint.h"
#include "replacement.h"


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
static int gmm_load(struct region **rgns, int nrgns);
static int gmm_launch(const char *entry, struct region **rgns, int nrgns);

// The GMM context for this process
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
	//pcontext->size_alloced = 0;
	pcontext->size_attached = 0;
	INIT_LIST_HEAD(&pcontext->list_alloced);
	initlock(&pcontext->lock_alloced);
	INIT_LIST_HEAD(&pcontext->list_attached);
	initlock(&pcontext->lock_attached);

	if (cudaStreamCreate(&pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("failed to create stream for dma\n");
		free(pcontext);
		pcontext = NULL;
		return -1;
	}

	if (cudaStreamCreate(&pcontext->stream_kernel) != cudaSuccess) {
		GMM_DPRINT("failed to create stream for kernel launch\n");
		cudaStreamDestroy(pcontext->stream_dma);
		free(pcontext);
		pcontext = NULL;
		return -1;
	}

	return 0;
}

void gmm_context_fini()
{
	// TODO: have to free all memory objects still attached to device

	cudaStreamDestroy(pcontext->stream_dma);
	cudaStreamDestroy(pcontext->stream_kernel);
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

// Which stream is the upcoming kernel to be issued to?
static cudaStream_t stream_issue = 0;

// TODO: Currently, %stream_issue is always set to pcontext->stream_kernel.
// This is not the best solution because it forbids kernels from being
// issued to different streams, which is required for, e.g., concurrent
// kernel executions.
// A better design is to prepare a kernel callback queue for each possible
// stream in pcontext->kcb; kernel callbacks are registered in queues where
// they are issued to. This both maintains the correctness of kernel callbacks
// and retains the capability that kernels being issued to multiple streams.
cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{
	stream_issue = pcontext->stream_kernel;
	return nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream_issue);
}

// Reference hints passed for a kernel launch
extern int refs[NREFS];
extern int int rwflags[NREFS];
extern int nrefs;

// The regions referenced by the following kernel to be launched
// TODO: should prepare the following structures for each stream
static struct dptr_arg dargs[NREFS];
static int nargs = 0;
static int iarg = 0;	// which argument current cudaSetupArgument refers to

// CUDA pushes kernel arguments from left to right. For example, for a kernel
//				k(a, b, c)
// , a will be pushed with cudaSetupArgument first, followed by b, and
// finally c. %offset gives the actual offset of an argument in the call
// stack, rather than which argument is being pushed.
//
// Let's assume cudaSetupArgument is invoked following the sequence of
// arguments.
cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{
	struct region *r;
	cudaError_t ret;
	int is_dptr = 0;
	int i;

	// Test whether this argument is a device memory pointer.
	// If it is, record it and postpone its pushing until cudaLaunch.
	// Use reference hints if given. Otherwise, parse automatically
	// (but parsing errors are possible, e.g., when the user pass the
	// long argument that happen to lay within some region's host swap
	// buffer area).
	// XXX: we should assume all memory objects are to be referenced
	// if not reference hints are given.
	if (nrefs > 0) {
		for (i = 0; i < nrefs; i++) {
			if (refs[i] == iarg)
				break;
		}
		if (i < nrefs) {
			if (size != sizeof(void *))
				panic("cudaSetupArgument does not match cudaReference");
			r = region_lookup(pcontext, *(void **)arg);
			if (!r)
				// TODO: report error more gracefully
				panic("region_lookup in cudaSetupArgument");
			is_dptr = 1;
		}
	}
	else if (size == sizeof(void *)) {
		r = region_lookup(pcontext, *(void **)arg);
		if (r)
			is_dptr = 1;
	}

	if (is_dptr) {
		dargs[nargs].r = r;
		dargs[nargs].off = (unsigned long)(*(void **)arg - r->addr_swp);
		if (nrefs > 0)
			dargs[nargs].flag = rwflags[i];
		else
			dargs[nargs].flag = 0;
		dargs[nargs++].argoff = offset;
		ret = cudaSuccess;
	}
	else
		// This argument is not a device memory pointer
		ret = nv_cudaSetupArgument(arg, size, offset);

	iarg++;
	return ret;
}

cudaError_t gmm_cudaLaunch(const char *entry)
{
	cudaError_t ret = cudaSuccess;
	struct region **rgns = NULL;
	int nrgns = 0;
	long totsize = 0;
	int i, j, shit;

	if (nrefs > NREFS)
		panic("nrefs");
	if (nargs <= 0 || nargs > NREFS)
		panic("nargs");

	// It is possible that multiple device pointers fall into
	// the same region. So we need to get the list of unique
	// regions referenced by the kernel being launched. Set
	// dynamic rw hints too.
	rgns = (struct region *)malloc(sizeof(*rgns) * NREFS);
	if (!rgns) {
		GMM_DPRINT("malloc failed for region array in gmm_cudaLaunch: %s\n", \
				strerror(errno));
		ret = cudaErrorLaunchOutOfResources;
		goto finish;
	}
	for (i = 0; i < nargs; i++) {
		if (!is_included(rgns, nrgns, dargs[i].r)) {
			rgns[nrgns++] = dargs[i].r;
			dargs[i].r->hint.rw_dynmic = dargs[i].flag;
			totsize += dargs[i].r->size;
		}
		else
			dargs[i].r->hint.rw_dynmic |= dargs[i].flag;
	}
	for (i = 0; i < nrgns; i++) {
		if (rgns[i]->hint.rw_dynmic == 0)
			rgns[i]->hint.rw_dynmic = rgns[i]->hint.rw_static;
	}

	if (totsize > get_memsize()) {
		GMM_DPRINT("kernel requires too much device memory space (%ld)\n", \
				totsize);
		free(rgns);
		ret = cudaErrorInvalidConfiguration;
		goto finish;
	}

	// Load all referenced regions. At any time, only one context
	// is allowed to load regions for kernel launch. This is to
	// avoid deadlocks/livelocks caused by memory contentions from
	// concurrent kernel launches.
	// TODO: Add kernel scheduling logic here. The order of loadings
	// plays an important role for fully overlapping kernel executions
	// and DMAs. Ideally, kernels with nothing to load should be issued
	// first. Then the loadings of other kernels can be overlapped with
	// kernel executions.
reload:
	begin_load();
	shit = gmm_load(rgns, nrgns);
	end_load();
	if (shit) {
		sched_yield();
		goto reload;
	}

	// Process RW hints.
	// TODO: the handling of RW hints needs to be re-organized. E.g.,
	// what if the launch below failed, how to specify partial
	// modification.
	for (i = 0; i < nrgns; i++) {
		if (rgns[i]->hint.rw_dynmic & HINT_WRITE) {
			region_inval(rgns[i], 1);
			region_valid(rgns[i], 0);
		}
	}

	// Push all device pointer arguments.
	for (i = 0; i < nargs; i++) {
		dargs[i].dptr = dargs[i].r->addr_dev + dargs[i].off;
		nv_cudaSetupArgument(&dargs[i].dptr, sizeof(void *), dargs[i].argoff);
	}

	// Now we can launch the kernel.
	if (gmm_launch(entry, rgns, nrgns) < 0) {
		for (i = 0; i < nrgns; i++)
			region_unpin(rgns[i]);
		ret = cudaErrorUnknown;
	}

finish:
	if (rgns)
		free(rgns);
	nrefs = 0;
	nargs = 0;
	iarg = 0;
	return ret;
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
	default: // STATE_DETACHED
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

// TODO: When to sync and how to sync needs to be re-thought thoroughly.
static int gmm_memcpy_dtoh(void *dst, void *src, unsigned long size)
{
	if (nv_cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
			pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("DtoH (%lu, %p => %p) failed\n", size, src, dst);
		return -1;
	}

	if (cudaStreamSynchronize(pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("Sync over DMA stream failed\n");
		return -1;
	}

	return 0;
}

// TODO: When to sync and how to sync needs to be re-thought thoroughly.
static int gmm_memcpy_htod(void *dst, void *src, unsigned long size)
{
	if (nv_cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
			pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("DtoH (%lu, %p => %p) failed\n", size, src, dst);
		return -1;
	}

	if (cudaStreamSynchronize(pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("Sync over DMA stream failed\n");
		return -1;
	}

	return 0;
}

// Sync the host and device copies of a data block.
// The direction of sync is determined by current valid flags. Data are synced
// from the valid copy to the invalid copy.
static void block_sync(struct region *r, int block)
{
	int dvalid = r->blocks[block].dev_valid;
	int svalid = r->blocks[block].swp_valid;
	unsigned long off1, off2;

	// Nothing to sync if both are valid or both are invalid
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
		struct region *r,
		void *dst,
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
		gmm_dtoh_block(r, dst, off, size, iblock, 1, skipped + iblock - ifirst);
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
			gmm_dtoh_block(r, dst, off, size, iblock, 0, NULL);
		dst += size;
		off += size;
		iblock++;
	}

	free(skipped);
	return 0;
}

// Select victims for %size_needed bytes of free device memory space.
// %excls[0:%nexcl) are local regions that should not be selected.
// Put selected victims in the list %victims.
int victim_select(
		long size_needed,
		struct region **excls,
		int nexcl,
		struct list_head *victims)
{
	int ret = 0;

#if defined(GMM_REPLACEMENT_LRU)
	ret = victim_select_lru(size_needed, excls, nexcl, victims);
#elif defined(GMM_REPLACEMENT_LFU)
	ret = victim_select_lfu(size_needed, excls, nexcl, victims);
#else
	panic("replacement policy not specified");
	ret = -1;
#endif

	return ret;
}

int region_evict(struct region *r)
{
	int i;

	for (i = 0; i < NRBLOCKS(r->size); i++) {
		if (!r->blocks[i].swp_valid)
			block_sync(r, i);
	}
	nv_cudaMemFree(r->addr_dev);
	r->addr_dev = NULL;

	list_attached_del(pcontext, r);
	r->state = STATE_DETACHED;
	return 0;
}

// Evict the victim %victim.
// %victim may point to a local region or a remote client that
// may own some evictable region.
int victim_evict(struct victim *victim, long size_needed)
{
	if (victim->r)
		return region_evict(victim->r);
	else
		return remote_victim_evict(victim->client,
				size_needed - get_free_memsize_signed());
}

// Evict some device memory so that the size of free space can
// satisfy %size_needed. Regions in %excls[0:%nexcl) should not
// be selected for eviction.
static int gmm_evict(long size_needed, struct region **excls, int nexcl)
{
	struct list_head victims, *e;
	struct victim *v;

	INIT_LIST_HEAD(&victims);

	do {
		if (victim_select(size_needed, excls, nexcl, &victims) < 0)
			return -1;
		for (e = victims.next; e != (&victims); ) {
			v = list_entry(e, struct victim, entry);
			if (get_free_memsize() < size_needed) {
				if (victim_evict(v, size_needed) < 0)
					goto fail_evict;
			}
			else if (v->r) {
				acquire(&v->r->lock);
				if (v->r->state != STATE_FREEING)
					v->r->state = STATE_ATTACHED;
				release(&v->r->lock);
			}
			else if (v->client != -1) {
				// TODO: unpin the remote client
			}
			list_del(e);
			e = e->next;
			free(v);
		}
	} while (get_free_memsize() < size_needed);

	return 0;

fail_evict:
	for (e = victims.next; e != (&victims); ) {
		v = list_entry(e, struct victim, entry);
		if (v->r) {
			acquire(&v->r->lock);
			if (v->r->state != STATE_FREEING)
				v->r->state = STATE_ATTACHED;
			release(&v->r->lock);
		}
		list_del(e);
		e = e->next;
		free(v);
	}

	return -1;
}

// Allocate device memory to a region (i.e., attach).
static int region_attach(
		struct region *r,
		int pin,
		struct region **excls,
		int nexcl)
{
	int i;

	if (r->state != STATE_DETACHED) {
		GMM_DPRINT("nothing to attach\n");
		return -1;
	}

	// Attach if current free memory space is larger than region size.
	if (r->size <= client_free_memsize() &&
		nv_cudaMalloc(&r->addr_dev, r->size) == cudaSuccess)
		goto attach_success;

	// Evict some device memory.
	if (gmm_evict(r->size, excls, nexcl) < 0 && r->size > get_free_memsize())
		return -1;

	// Try to attach again.
	if (nv_cudaMalloc(&r->addr_dev, r->size) != cudaSuccess) {
		r->addr_dev = NULL;
		return -1;
	}

attach_success:
	update_attached(r->size);
	atomic_addl(&pcontext->size_attached, r->size);
	if (pin)
		region_pin(r);
	// Reassure that the dev copies of all blocks are set to invalid
	region_inval(r, 0);
	r->state = STATE_ATTACHED;
	list_attached_add(pcontext, r);

	return 0;
}

// Load a region to device memory.
// excls[0:nexcl) are regions that should not be evicted when
// evictions need to happen during the loading.
static int region_load(
		struct region *r,
		int pin,
		struct region **excls,
		int nexcl)
{
	int i;

	if (r->state == STATE_EVICTING || r->state == STATE_FREEING) {
		GMM_DPRINT("should not see a evicting/freeing region during loading\n");
		return -1;
	}

	// Attach if the region is still detached
	if (r->state == STATE_DETACHED)
		if (region_attach(r, 1, excls, nexcl) == -1)
			return -1;
	else
		region_pin(r);

	// Fetch data to device memory if necessary
	if (r->hint.rw_dynmic & HINT_READ) {
		for (i = 0; i < NRBLOCKS(r->size); i++) {
			if (!r->blocks[i].dev_valid)
				block_sync(r, i);
		}
	}

	return 0;
}

// Load all %n regions specified by %rgns to device.
// Every successfully loaded region is pinned to device.
// If all regions cannot be loaded successfully, successfully
// loaded regions will be unpinned so that they can be
// replaced by other kernel launches.
static int gmm_load(struct region **rgns, int n)
{
	char *pinned;
	int i, ret;

	if (!rgns || n <= 0)
		return -1;

	pinned = (char *)malloc(n);
	if (!pinned) {
		GMM_DPRINT("malloc failed for pinned array: %s\n", strerr(errno));
		return -1;
	}
	memset(pinned, 0, n);

	for (i = 0; i < n; i++) {
		if (rgns[i]->state == STATE_FREEING) {
			GMM_DPRINT("warning: not loading freed region\n");
			continue;
		}
		// NOTE: In current design, this locking is redundant
		acquire(&rgns[i]->lock);
		ret = region_load(rgns[i], 1, rgns, n);
		release(&rgns[i]->lock);
		if (ret < 0)
			goto fail;
		pinned[i] = 1;
	}

	free(pinned);
	return 0;

fail:
	for (i = 0; i < n; i++)
		if (pinned[i])
			region_unpin(rgns[i]);
	free(pinned);
	return ret;
}

// The callback function invoked by CUDA after each kernel finishes
// execution. Have to keep it as short as possible because it blocks
// the following commands in the stream.
void CUDART_CB gmm_kernel_callback(
		cudaStream_t stream,
		cudaError_t status,
		void *data)
{
	struct kcb *pcb = (struct kcb *)data;
	int i;
	for (i = 0; i < pcb->nrgns; i++)
		region_unpin(pcb->rgns[i]);
	free(pcb);
}

// Here we utilize CUDA 5.0's stream callback feature to capture kernel
// finish event and unpin related regions accordingly.
static int gmm_launch(const char *entry, struct region **rgns, int nrgns)
{
	struct kcb *pcb;

	if (nrgns > NREFS) {
		GMM_DPRINT("too many regions\n");
		return -1;
	}

	pcb = (struct kcb *)malloc(sizeof(*pcb));
	if (!pcb) {
		GMM_DPRINT("malloc failed for kcb: %s\n", strerr(errno));
		return -1;
	}
	memcpy(&pcb->rgns, rgns, sizeof(void *) * nrgns);
	pcb->nrgns = nrgns;

	if (nv_cudaLaunch(entry) != cudaSuccess) {
		free(pcb);
		return -1;
	}
	cudaStreamAddCallback(stream_issue, gmm_kernel_callback, (void *)pcb, 0);

	return 0;
}
