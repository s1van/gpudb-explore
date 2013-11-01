#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sched.h>

#include "common.h"
#include "client.h"
#include "core.h"
#include "hint.h"
#include "replacement.h"
#include "msq.h"
#include "debug.h"


// CUDA function handlers, defined in gmm_interfaces.c
extern cudaError_t (*nv_cudaMalloc)(void **, size_t);
extern cudaError_t (*nv_cudaFree)(void *);
//extern cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t,
//		enum cudaMemcpyKind);
extern cudaError_t (*nv_cudaMemcpyAsync)(void *, const void *,
		size_t, enum cudaMemcpyKind, cudaStream_t stream);
extern cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *);
extern cudaError_t (*nv_cudaStreamDestroy)(cudaStream_t);
extern cudaError_t (*nv_cudaStreamSynchronize)(cudaStream_t);
extern cudaError_t (*nv_cudaMemGetInfo)(size_t*, size_t*);
extern cudaError_t (*nv_cudaSetupArgument) (const void *, size_t, size_t);
extern cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
extern cudaError_t (*nv_cudaMemset)(void * , int , size_t );
//extern cudaError_t (*nv_cudaMemsetAsync)(void * , int , size_t, cudaStream_t);
//extern cudaError_t (*nv_cudaDeviceSynchronize)(void);
extern cudaError_t (*nv_cudaLaunch)(const void *);
extern cudaError_t (*nv_cudaStreamAddCallback)(cudaStream_t,
		cudaStreamCallback_t, void*, unsigned int);

static int gmm_free(struct region *m);
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
static int gmm_dtod(
		struct region *rd,
		struct region *rs,
		void *dst,
		const void *src,
		size_t count);
static int gmm_memset(struct region *r, void *dst, int value, size_t count);
static int gmm_load(struct region **rgns, int nrgns);
static int gmm_launch(const char *entry, struct region **rgns, int nrgns);
struct region *region_lookup(struct gmm_context *ctx, const void *ptr);
static int block_sync(struct region *r, int block);


// The GMM context for this process
struct gmm_context *pcontext = NULL;


static void list_alloced_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_add(&r->entry_alloced, &ctx->list_alloced);
	release(&ctx->lock_alloced);
}

static void list_alloced_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_del(&r->entry_alloced);
	release(&ctx->lock_alloced);
}

static void list_attached_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_add(&r->entry_attached, &ctx->list_attached);
	release(&ctx->lock_attached);
}

static void list_attached_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_del(&r->entry_attached);
	release(&ctx->lock_attached);
}

static void list_attached_mov(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_move(&r->entry_attached, &ctx->list_attached);
	release(&ctx->lock_attached);
}


static inline void region_pin(struct region *r)
{
	int pinned = atomic_inc(&(r)->pinned);
	if (pinned == 0)
		update_detachable(-r->size);
}

static inline void region_unpin(struct region *r)
{
	int pinned = atomic_dec(&(r)->pinned);
	if (pinned == 1)
		update_detachable(r->size);
}

// Initialize local GMM context.
int gmm_context_init()
{
	if (pcontext != NULL) {
		gprint(FATAL, "pcontext already exists!\n");
		return -1;
	}

	pcontext = (struct gmm_context *)malloc(sizeof(*pcontext));
	if (!pcontext) {
		gprint(FATAL, "malloc failed for pcontext: %s\n", strerror(errno));
		return -1;
	}

	initlock(&pcontext->lock);		// ???
	latomic_set(&pcontext->size_attached, 0L);
	INIT_LIST_HEAD(&pcontext->list_alloced);
	INIT_LIST_HEAD(&pcontext->list_attached);
	initlock(&pcontext->lock_alloced);
	initlock(&pcontext->lock_attached);

	if (nv_cudaStreamCreate(&pcontext->stream_dma) != cudaSuccess) {
		gprint(FATAL, "failed to create DMA stream\n");
		free(pcontext);
		pcontext = NULL;
		return -1;
	}

	if (nv_cudaStreamCreate(&pcontext->stream_kernel) != cudaSuccess) {
		gprint(FATAL, "failed to create kernel stream\n");
		nv_cudaStreamDestroy(pcontext->stream_dma);
		free(pcontext);
		pcontext = NULL;
		return -1;
	}

	return 0;
}

void gmm_context_fini()
{
	struct list_head *p;
	struct region *r;

	// Free all dangling memory regions.
	while (!list_empty(&pcontext->list_alloced)) {
		p = pcontext->list_alloced.next;
		r = list_entry(p, struct region, entry_alloced);
		if (!gmm_free(r))
			list_move_tail(p, &pcontext->list_alloced);
	}

	nv_cudaStreamDestroy(pcontext->stream_dma);
	nv_cudaStreamDestroy(pcontext->stream_kernel);
	free(pcontext);
	pcontext = NULL;
}

// Allocate a new device memory object.
// We only allocate the host swap buffer space for now, and return
// the address of the host buffer to the user as the identifier of
// the object.
cudaError_t gmm_cudaMalloc(void **devPtr, size_t size, int flags)
{
	struct region *r;
	int nblocks;

	gprint(DEBUG, "cudaMalloc begins: size(%lu) flags(%x)\n", size, flags);

	if (size > memsize_total()) {
		gprint(ERROR, "cudaMalloc size (%lu) too large (max %ld)\n", \
				size, memsize_total());
		return cudaErrorInvalidValue;
	}
	else if (size <= 0) {
		gprint(ERROR, "cudaMalloc size (%lu) too small\n", size);
		return cudaErrorInvalidValue;
	}

	r = (struct region *)calloc(1, sizeof(*r));
	if (!r) {
		gprint(FATAL, "malloc for a new region: %s\n", strerror(errno));
		return cudaErrorMemoryAllocation;
	}

	r->swp_addr = malloc(size);
	if (!r->swp_addr) {
		gprint(FATAL, "malloc failed for swap buffer: %s\n", strerror(errno));
		free(r);
		return cudaErrorMemoryAllocation;
	}

	if (flags & HINT_PTARRAY) {
		if (size % sizeof(void *)) {
			gprint(ERROR, "dptr array size (%lu) not aligned\n", size);
			free(r->swp_addr);
			free(r);
			return cudaErrorInvalidValue;
		}
		r->pta_addr = calloc(1, size);
		if (!r->pta_addr) {
			gprint(FATAL, "malloc failed for dptr array: %s\n", \
					strerror(errno));
			free(r->swp_addr);
			free(r);
			return cudaErrorMemoryAllocation;
		}
		gprint(DEBUG, "pta_addr (%p) alloced for %p\n", r->pta_addr, r);
	}

	nblocks = NRBLOCKS(size);
	r->blocks = (struct block *)calloc(nblocks, sizeof(struct block));
	if (!r->blocks) {
		gprint(FATAL, "malloc failed for blocks array: %s\n", strerror(errno));
		if (r->pta_addr)
			free(r->pta_addr);
		free(r->swp_addr);
		free(r);
		return cudaErrorMemoryAllocation;
	}

	// TODO: test how CUDA runtime aligns the size of memory allocations
	r->size = (long)size;
	//initlock(&r->lock);
	r->state = STATE_DETACHED;
	//atomic_set(&r->pinned, 0);
	r->flags = flags;
	//r->rwhint.flags = HINT_DEFAULT;

	list_alloced_add(pcontext, r);
	*devPtr = r->swp_addr;

	gprint(DEBUG, "cudaMalloc ends: r(%p) swp(%p) pta(%p)\n", \
			r, r->swp_addr, r->pta_addr);
	return cudaSuccess;
}

cudaError_t gmm_cudaFree(void *devPtr)
{
	struct region *r;

	if (!(r = region_lookup(pcontext, devPtr))) {
		gprint(ERROR, "cannot find region containing %p in cudaFree\n", devPtr);
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
		gprint(ERROR, "cannot find region containing %p in htod\n", dst);
		return cudaErrorInvalidDevicePointer;
	}
	if (r->state == STATE_FREEING || r->state == STATE_ZOMBIE) {
		gprint(ERROR, "region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (dst + count > r->swp_addr + r->size) {
		gprint(ERROR, "htod out of region boundary\n");
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
		gprint(ERROR, "cannot find region containing %p in dtoh\n", src);
		return cudaErrorInvalidDevicePointer;
	}
	if (r->state == STATE_FREEING || r->state == STATE_ZOMBIE) {
		gprint(ERROR, "region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (src + count > r->swp_addr + r->size) {
		gprint(ERROR, "dtoh out of region boundary\n");
		return cudaErrorInvalidValue;
	}

	if (gmm_dtoh(r, dst, src, count) < 0)
		return cudaErrorUnknown;

	return cudaSuccess;
}

cudaError_t gmm_cudaMemcpyDtoD(
		void *dst,
		const void *src,
		size_t count)
{
	struct region *rs, *rd;

	if (count <= 0)
		return cudaErrorInvalidValue;

	rs = region_lookup(pcontext, src);
	if (!rs) {
		gprint(ERROR, "cannot find src region containing %p in dtod\n", src);
		return cudaErrorInvalidDevicePointer;
	}
	if (rs->state == STATE_FREEING || rs->state == STATE_ZOMBIE) {
		gprint(ERROR, "src region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (src + count > rs->swp_addr + rs->size) {
		gprint(ERROR, "dtod out of src boundary\n");
		return cudaErrorInvalidValue;
	}

	rd = region_lookup(pcontext, dst);
	if (!rd) {
		gprint(ERROR, "cannot find dst region containing %p in dtod\n", dst);
		return cudaErrorInvalidDevicePointer;
	}
	if (rd->state == STATE_FREEING || rd->state == STATE_ZOMBIE) {
		gprint(ERROR, "dst region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (dst + count > rd->swp_addr + rd->size) {
		gprint(ERROR, "dtod out of dst boundary\n");
		return cudaErrorInvalidValue;
	}

	if (gmm_dtod(rd, rs, dst, src, count) < 0)
		return cudaErrorUnknown;

	return cudaSuccess;
}

cudaError_t gmm_cudaMemGetInfo(size_t *free, size_t *total)
{
	*free = (size_t)memsize_free();
	*total = (size_t)memsize_total();
	gprint(DEBUG, "cudaMemGetInfo: free(%lu) total(%lu)\n", *free, *total);
	return cudaSuccess;
}

// Reference hints passed for a kernel launch.
// Set by cudaReference() in interfaces.c.
// TODO: have to reset nrefs in case of errors before cudaLaunch.
extern int refs[NREFS];
extern int rwflags[NREFS];
extern int nrefs;

// The arguments for the following kernel to be launched.
// TODO: should prepare the following structures for each stream.
static unsigned char kstack[512];		// Temporary kernel argument stack
static void *ktop = (void *)kstack;		// Stack top
static struct karg kargs[NREFS];
static int nargs = 0;

// Which stream is the upcoming kernel to be issued to?
static cudaStream_t stream_issue = 0;

// TODO: Currently, %stream_issue is always set to pcontext->stream_kernel.
// This is not the best solution because it forbids kernels from being
// issued to different streams, which is required for, e.g., concurrent
// kernel executions.
// A better design is to prepare a kernel callback queue in pcontext->kcb
// for each possible stream ; kernel callbacks are registered in queues where
// they are issued to. This both maintains the correctness of kernel callbacks
// and retains the capability that kernels being issued to multiple streams.
cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{
	gprint(DEBUG, "cudaConfigureCall: %d %d %lu\n", \
			gridDim.x, blockDim.x, sharedMem);
	nargs = 0;
	ktop = (void *)kstack;
	stream_issue = pcontext->stream_kernel;
	return nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream_issue);
}

// CUDA pushes kernel arguments from left to right. For example, for a kernel
//				k(a, b, c)
// , a will be pushed on top of the stack, followed by b, and finally c.
// %offset gives the actual offset of an argument in the call stack,
// rather than which argument is being pushed. %size is the size of the
// argument being pushed. Note that argument offset will be aligned based
// on the type of argument (e.g., a (void *)-type argument's offset has to
// be aligned to sizeof(void *)).
cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{
	struct region *r;
	int is_dptr = 0;
	int i = 0;

	gprint(DEBUG, "cudaSetupArgument: nargs(%d) size(%lu) offset(%lu)\n", \
			nargs, size, offset);

	// Test whether this argument is a device memory pointer.
	// If it is, record it and postpone its pushing until cudaLaunch.
	// Use reference hints if given. Otherwise, parse automatically
	// (but parsing errors are possible, e.g., when the user passes a
	// long argument that happen to lay within some region's host swap
	// buffer area).
	if (nrefs > 0) {
		for (i = 0; i < nrefs; i++) {
			if (refs[i] == nargs)
				break;
		}
		if (i < nrefs) {
			if (size != sizeof(void *)) {
				gprint(ERROR, "argument size (%lu) does not match " \
						"cudaReference (%d)\n", size, nargs);
				return cudaErrorUnknown;
				//panic("cudaSetupArgument does not match cudaReference");
			}
			r = region_lookup(pcontext, *(void **)arg);
			if (!r) {
				gprint(ERROR, "cannot find region containing %p (%d) in " \
						"cudaSetupArgument\n", *(void **)arg, nargs);
				return cudaErrorUnknown;
				//panic("region_lookup in cudaSetupArgument");
			}
			is_dptr = 1;
		}
	}
	// TODO: we should assume all memory regions are to be referenced
	// if no reference hints are given.
	else if (size == sizeof(void *)) {
		r = region_lookup(pcontext, *(void **)arg);
		if (r)
			is_dptr = 1;
	}

	if (is_dptr) {
		kargs[nargs].arg.arg1.r = r;
		kargs[nargs].arg.arg1.off =
				(unsigned long)(*(void **)arg - r->swp_addr);
		if (nrefs > 0)
			kargs[nargs].arg.arg1.flags = rwflags[i];
		else
			kargs[nargs].arg.arg1.flags = HINT_DEFAULT | HINT_PTADEFAULT;
		gprint(DEBUG, "argument is dptr: r(%p %p %ld %d %d)\n", \
				r, r->swp_addr, r->size, r->flags, r->state);
	}
	else {
		// This argument is not a device memory pointer.
		// XXX: Currently we ignore the case that nv_cudaSetupArgument
		// returns error and CUDA runtime might stop pushing arguments.
		memcpy(ktop, arg, size);
		kargs[nargs].arg.arg2.arg = ktop;
		ktop += size;
	}
	kargs[nargs].is_dptr = is_dptr;
	kargs[nargs].size = size;
	kargs[nargs].argoff = offset;

	nargs++;
	return cudaSuccess;
}

// TODO: We should assume that all memory regions are referenced
// if no reference hints are given.
// I.e., if (nargs > 0) do the following; else add all regions.
static long regions_referenced(struct region ***prgns, int *pnrgns)
{
	struct region **rgns, *r;
	long total = 0;
	int nrgns = 0;
	int i;

	if (nrefs > NREFS)
		panic("nrefs");
	if (nargs <= 0 || nargs > NREFS)
		panic("nargs");

	// Get the upper bound of the number of unique regions.
	for (i = 0; i < nargs; i++) {
		if (kargs[i].is_dptr) {
			nrgns++;
			r = kargs[i].arg.arg1.r;
			// Here we assume at most one level of dptr arrays
			if (r->flags & HINT_PTARRAY)
				nrgns += r->size / sizeof(void *);
		}
	}
	if (nrgns <= 0)
		panic("nrgns");

	rgns = (struct region **)malloc(sizeof(*rgns) * nrgns);
	if (!rgns) {
		gprint(FATAL, "malloc failed for region array: %s\n", strerror(errno));
		return -1;
	}
	nrgns = 0;

	// Now set the regions to be referenced.
	for (i = 0; i < nargs; i++) {
		if (kargs[i].is_dptr) {
			r = kargs[i].arg.arg1.r;
			if (!is_included((void **)rgns, nrgns, (void*)r)) {
				rgns[nrgns++] = r;
				r->rwhint.flags = kargs[i].arg.arg1.flags & HINT_MASK;
				total += r->size;

				if (r->flags & HINT_PTARRAY) {
					void **pdptr = (void **)(r->pta_addr);
					void **pend = (void **)(r->pta_addr + r->size);
					r->rwhint.flags &= ~HINT_WRITE;	// dptr array is read-only
					// For each device memory pointer contained in this region
					while (pdptr < pend) {
						r = region_lookup(pcontext, *pdptr);
						if (!r) {
							gprint(WARN, "cannot find region for dptr " \
									"%p (%d)\n", *pdptr, i);
							pdptr++;
							continue;
						}
						if (!is_included((void **)rgns, nrgns, (void*)r)) {
							rgns[nrgns++] = r;
							r->rwhint.flags =
									((kargs[i].arg.arg1.flags & HINT_PTAREAD) ?
											HINT_READ : 0) |
									((kargs[i].arg.arg1.flags & HINT_PTAWRITE) ?
											HINT_WRITE : 0);
							total += r->size;
						}
						else
							r->rwhint.flags |=
									((kargs[i].arg.arg1.flags & HINT_PTAREAD) ?
											HINT_READ : 0) |
									((kargs[i].arg.arg1.flags & HINT_PTAWRITE) ?
											HINT_WRITE : 0);
						pdptr++;
					}
				}
			}
			else
				r->rwhint.flags |= kargs[i].arg.arg1.flags & HINT_MASK;
		}
	}

	*pnrgns = nrgns;
	if (nrgns > 0)
		*prgns = rgns;
	else {
		free(rgns);
		*prgns = NULL;
	}

	return total;
}

// Priority of the kernel launch (defined in interfaces.c).
// TODO: have to arrange something in global shared memory
// to expose kernel launch scheduling info.
extern int prio_kernel;

// It is possible that multiple device pointers fall into
// the same region. So we first need to get the list of
// unique regions referenced by the kernel being launched.
// Then, load all referenced regions. At any time, only one
// context is allowed to load regions for kernel launch.
// This is to avoid deadlocks/livelocks caused by memory
// contentions from simultaneous kernel launches.
//
// TODO: Add kernel scheduling logic. The order of loadings
// plays an important role for fully overlapping kernel executions
// and DMAs. Ideally, kernels with nothing to load should be issued
// first. Then the loadings of other kernels can be overlapped with
// kernel executions.
// TODO: Maybe we should allow multiple loadings to happen
// simultaneously if we know that free memory is enough.
cudaError_t gmm_cudaLaunch(const char *entry)
{
	cudaError_t ret = cudaSuccess;
	struct region **rgns = NULL;
	int nrgns = 0;
	long total = 0;
	int i, ldret;

	gprint(DEBUG, "cudaLaunch\n");

	// NOTE: it is possible that nrgns == 0 when regions_referenced
	// returns. Consider a kernel that only uses registers, for
	// example.
	total = regions_referenced(&rgns, &nrgns);
	if (total < 0) {
		gprint(ERROR, "failed to get the regions to be referenced\n");
		ret = cudaErrorUnknown;
		goto finish;
	}
	else if (total > memsize_total()) {
		gprint(ERROR, "kernel requires too much space (%ld)\n", total);
		ret = cudaErrorInvalidConfiguration;
		goto finish;
	}

reload:
	launch_wait();
	ldret = gmm_load(rgns, nrgns);
	launch_signal();
	if (ldret > 0) {	// load unsuccessful, retry later
		sched_yield();
		goto reload;
	}
	else if (ldret < 0) {	// fatal load error, quit launching
		gprint(ERROR, "load failed; quitting kernel launch\n");
		ret = cudaErrorUnknown;
		goto finish;
	}

	// Process WRITE hints and transfer the real dptrs to device
	// memory for dptr arrays.
	// By this moment, all regions pointed by each dptr array has
	// been loaded and pinned to device memory.
	// XXX: What if the launch below failed? Partial modification?
	for (i = 0; i < nrgns; i++) {
		if (rgns[i]->rwhint.flags & HINT_WRITE) {
			region_inval(rgns[i], 1);
			region_valid(rgns[i], 0);
		}

		if (rgns[i]->flags & HINT_PTARRAY) {
			void **pdptr = (void **)(rgns[i]->pta_addr);
			void **pend = (void **)(rgns[i]->pta_addr + rgns[i]->size);
			unsigned long off = 0;
			int j;

			while (pdptr < pend) {
				struct region *r = region_lookup(pcontext, *pdptr);
				if (!r) {
					gprint(WARN, "cannot find region for dptr " \
							"%p (%d)\n", *pdptr, i);
					off += sizeof(void *);
					pdptr++;
					continue;
				}
				*(void **)(rgns[i]->swp_addr + off) = r->dev_addr +
						(unsigned long)(*pdptr - r->swp_addr);
				off += sizeof(void *);
				pdptr++;
			}

			region_valid(rgns[i], 1);
			region_inval(rgns[i], 0);
			for (j = 0; j < NRBLOCKS(rgns[i]->size); j++) {
				ret = block_sync(rgns[i], j);
				if (ret != 0) {
					for (i = 0; i < nrgns; i++)
						region_unpin(rgns[i]);
					ret = cudaErrorUnknown;
					goto finish;
				}
			}
		}
	}

	// Push all kernel arguments.
	// TODO: create temporary argument array.
	for (i = 0; i < nargs; i++) {
		if (kargs[i].is_dptr) {
			kargs[i].arg.arg1.dptr =
					kargs[i].arg.arg1.r->dev_addr + kargs[i].arg.arg1.off;
			nv_cudaSetupArgument(&kargs[i].arg.arg1.dptr,
					kargs[i].size, kargs[i].argoff);
			/*gprint(DEBUG, "setup %p %lu %lu\n", \
					&kargs[i].arg.arg1.dptr, \
					sizeof(void *), \
					kargs[i].arg.arg1.argoff);*/
		}
		else {
			nv_cudaSetupArgument(kargs[i].arg.arg2.arg,
					kargs[i].size, kargs[i].argoff);
			/*gprint(DEBUG, "setup %p %lu %lu\n", \
					kargs[i].arg.arg2.arg, \
					kargs[i].arg.arg2.size, \
					kargs[i].arg.arg2.offset);*/
		}
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
	ktop = (void *)kstack;
	return ret;
}

cudaError_t gmm_cudaMemset(void *devPtr, int value, size_t count)
{
	struct region *r;

	if (count <= 0)
		return cudaErrorInvalidValue;

	r = region_lookup(pcontext, devPtr);
	if (!r) {
		gprint(ERROR, "cannot find region containing %p in cudaMemset\n", \
				devPtr);
		return cudaErrorInvalidDevicePointer;
	}
	if (r->state == STATE_FREEING || r->state == STATE_ZOMBIE) {
		gprint(ERROR, "region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (devPtr + count > r->swp_addr + r->size) {
		gprint(ERROR, "cudaMemset out of region boundary\n");
		return cudaErrorInvalidValue;
	}

	if (gmm_memset(r, devPtr, value, count) < 0)
		return cudaErrorUnknown;

	return cudaSuccess;
}

// The return value of this function tells whether the region has been
// immediately freed. 0 - not freed yet; 1 - freed.
static int gmm_free(struct region *r)
{
	gprint(DEBUG, "freeing r(%p %p %ld %d %d)\n", \
			r, r->swp_addr, r->size, r->flags, r->state);

	// First, properly inspect/set region state.
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
		gprint(DEBUG, "region set freeing\n");
		return 0;
	case STATE_FREEING:
		// The evictor has not seen this region being freed
		release(&r->lock);
		sched_yield();
		goto re_acquire;
	default: // STATE_DETACHED or STATE_ZOMBIE
		break;
	}
	release(&r->lock);

	// Now, this memory region can be freed.
	list_alloced_del(pcontext, r);
	if (r->blocks)
		free(r->blocks);
	if (r->pta_addr)
		free(r->pta_addr);
	if (r->swp_addr)
		free(r->swp_addr);
	if (r->dev_addr) {
		nv_cudaFree(r->dev_addr);
		latomic_sub(&pcontext->size_attached, r->size);
		update_attached(-r->size);
		update_detachable(-r->size);
	}
	free(r);

	gprint(DEBUG, "region freed\n");
	return 1;
}

// TODO: provide two implementations - sync and async.
// For async, use streamcallback to unpin if necessary.
// TODO: use host pinned buffer.
static int gmm_memcpy_dtoh(void *dst, const void *src, unsigned long size)
{
	cudaError_t error; //= cudaGetLastError();

	if ((error = nv_cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
			pcontext->stream_dma)) != cudaSuccess) {
		gprint(ERROR, "DtoH (%lu, %p => %p) failed: %s (%d)\n", \
				size, src, dst, cudaGetErrorString(error), error);
		return -1;
	}

	if (nv_cudaStreamSynchronize(pcontext->stream_dma) != cudaSuccess) {
		gprint(ERROR, "Sync over DMA stream failed\n");
		return -1;
	}

	return 0;
}

// TODO: sync and async.
// TODO: use host pinned buffer.
static int gmm_memcpy_htod(void *dst, const void *src, unsigned long size)
{
	cudaError_t error; //= cudaGetLastError();

	if ((error = nv_cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
			pcontext->stream_dma)) != cudaSuccess) {
		gprint(ERROR, "DtoH (%lu, %p => %p) failed: %s (%d)\n", \
				size, src, dst, cudaGetErrorString(error), error);
		return -1;
	}

	if (nv_cudaStreamSynchronize(pcontext->stream_dma) != cudaSuccess) {
		gprint(ERROR, "Sync over DMA stream failed\n");
		return -1;
	}

	return 0;
}

// Sync the host and device copies of a data block.
// The direction of sync is determined by current valid flags. Data are synced
// from the valid copy to the invalid copy.
// NOTE:
// Block syncing has to ensure data consistency: a block being modified in a
// kernel cannot be synced from host to device or vice versa until the kernel
// finishes; a block being read in a kernel cannot be synced from host to
// device until the kernel finishes.
static int block_sync(struct region *r, int block)
{
	int dvalid = r->blocks[block].dev_valid;
	int svalid = r->blocks[block].swp_valid;
	unsigned long off, size;
	int ret = 0;

	// Nothing to sync if both are valid or both are invalid
	if ((dvalid ^ svalid) == 0)
		return 0;
	if (!r->dev_addr || !r->swp_addr)
		panic("block_sync");

	gprint(DEBUG, \
			"block sync begins: r(%p) block(%d) svalid(%d) dvalid(%d)\n", \
			r, block, svalid, dvalid);

	// Have to wait until the kernel modifying the region finishes,
	// otherwise it is possible that the data we read are inconsistent
	// with what's being written by the kernel.
	// TODO: will make the modifying flags more fine-grained (block level).
	while (atomic_read(&r->writing) > 0) ;

	off = block * BLOCKSIZE;
	size = MIN(off + BLOCKSIZE, r->size) - off;
	if (dvalid && !svalid) {
		// Sync from device to host swap buffer
		ret = gmm_memcpy_dtoh(r->swp_addr + off, r->dev_addr + off, size);
		if (!ret)
			r->blocks[block].swp_valid = 1;
	}
	else {
		// Sync from host swap buffer to device
		while (atomic_read(&r->reading) > 0) ;
		ret = gmm_memcpy_htod(r->dev_addr + off, r->swp_addr + off, size);
		if (!ret)
			r->blocks[block].dev_valid = 1;
	}

	gprint(DEBUG, "block sync ends\n");
	return ret;
}

// Copy a piece of data from $src to (the host swap buffer of) a block in $r.
// $offset gives the offset of the destination address relative to $r->swp_addr.
// $size is the size of data to be copied.
// $block tells which block is being modified.
// $skip specifies whether to skip copying if the block is being locked.
// $skipped, if not null, returns whether skipped (1 - skipped; 0 - not skipped)
#ifdef GMM_CONFIG_HTOD_RADICAL
// This is an implementation matching the radical version of gmm_htod.
// TODO: this is not correct. see the non-radical version for reasons and
// fix
static int gmm_htod_block(
		struct region *r,
		unsigned long offset,
		const void *src,
		unsigned long size,
		int block,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + block;
	int ret = 0;

	// partial modification
	if ((offset % BLOCKSIZE) || (size < BLOCKSIZE && offset + size < r->size)) {
		if (b->swp_valid || !b->dev_valid) {
			// no locking needed
			memcpy(r->swp_addr + offset, src, size);
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
					return 0;
				}
			}
			if (b->swp_valid || !b->dev_valid) {
				release(&r->blocks[block].lock);
				memcpy(r->swp_addr + offset, src, size);
				if (!b->swp_valid)
					b->swp_valid = 1;
				if (b->dev_valid)
					b->dev_valid = 0;
			}
			else {
				// XXX: We don't need to pin the device memory because we are
				// holding the lock of a swp_valid=0,dev_valid=1 block, which
				// will prevent the evictor, if any, from freeing the device
				// memory under us.
				ret = block_sync(r, block);
				release(&b->lock);
				if (ret != 0)
					goto finish;
				memcpy(r->swp_addr + offset, src, size);
				b->dev_valid = 0;
			}
		}
	}
	// Full over-writing (its valid flags have been set in advance).
	else {
		while (!try_acquire(&b->lock)) {
			if (skip) {
				if (skipped)
					*skipped = 1;
				return;
			}
		}
		// Acquire the lock and release immediately, to avoid data races with
		// the evictor who happens to be writing the swp buffer.
		release(&b->lock);
		memcpy(r->swp_addr + offset, src, size);
	}

finish:
	if (skipped)
		*skipped = 0;
	return ret;
}
#else
// This is an implementation matching the conservative version of gmm_htod.
static int gmm_htod_block(
		struct region *r,
		unsigned long offset,
		const void *src,
		unsigned long size,
		int block,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + block;
	int partial = (offset % BLOCKSIZE) ||
			(size < BLOCKSIZE && (offset + size) < r->size);
	int ret = 0;

/*	GMM_DPRINT("gmm_htod_block: r(%p) offset(%lu) src(%p)" \
			" size(%lu) block(%d) partial(%d)\n", \
			r, offset, src, size, block, partial);
*/
	// This `no-locking' case will cause a subtle block sync problem:
	// Suppose this block is invalid in both swp and dev, then the
	// following actions will set its swp to valid. Now if the evictor
	// sees invalid swp before it being set to valid and decides to do
	// a block sync, it may accidentally sync the data from host to
	// device, which should never happen during a block eviction. So
	// the safe action is to only test/change the state of a block while
	// holding its lock.
//	if (b->swp_valid || !b->dev_valid) {
//		// no locking needed
//		memcpy(r->swp_addr + offset, src, size);
//		if (!b->swp_valid)
//			b->swp_valid = 1;
//		if (b->dev_valid)
//			b->dev_valid = 0;
//	}
//	else {

	while (!try_acquire(&b->lock)) {
		if (skip) {
			if (skipped)
				*skipped = 1;
			return 0;
		}
	}
	if (b->swp_valid || !b->dev_valid) {
		if (!b->swp_valid)
			b->swp_valid = 1;
		if (b->dev_valid)
			b->dev_valid = 0;
		release(&b->lock);
		// this is not thread-safe; otherwise, move memcpy before release
		memcpy(r->swp_addr + offset, src, size);
	}
	else { // dev_valid == 1 && swp_valid == 0
		if (partial) {
			// XXX: We don't need to pin the device memory because we are
			// holding the lock of a swp_valid=0,dev_valid=1 block, which
			// will prevent the evictor, if any, from freeing the device
			// memory under us.
			ret = block_sync(r, block);
			if (!ret)
				b->dev_valid = 0;
			release(&b->lock);
			if (ret != 0)
				goto finish;
		}
		else {
			b->swp_valid = 1;
			b->dev_valid = 0;
			release(&b->lock);
		}
		memcpy(r->swp_addr + offset, src, size);
	}

//	}

finish:
	if (skipped)
		*skipped = 0;
	return ret;
}
#endif

// Device pointer array regions are special. Their host swap buffers and
// device memory buffers are temporary, and are only meaningful right before
// and during a kernel execution. The opaque dptr values are stored in
// pta_addr, and are modified by the host program only.
static int gmm_htod_pta(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	unsigned long off = (unsigned long)(dst - r->swp_addr);

	gprint(DEBUG, "htod_pta\n");

	if (off % sizeof(void *)) {
		gprint(ERROR, "offset (%lu) not aligned for host to pta memcpy\n", off);
		return -1;
	}
	if (count % sizeof(void *)) {
		gprint(ERROR, "count (%lu) not aligned for host to pta memcpy\n", count);
		return -1;
	}

	memcpy(r->pta_addr + off, src, count);
	return 0;
}

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
	void *s = (void *)src;
	char *skipped;
	int ret = 0;

	if (r->flags & HINT_PTARRAY)
		return gmm_htod_pta(r, dst, src, count);

	off = (unsigned long)(dst - r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// For each full-block over-writing, set dev_valid=0 and swp_valid=1.
	// Since we know the memory range being over-written, setting flags ahead
	// help prevent the evictor, if there is one, from wasting time evicting
	// those blocks. This is one unique advantage of us compared with CPU
	// memory management, where the OS usually does not have such interfaces
	// or knowledge.
	if (ifirst == ilast &&
		(count == BLOCKSIZE || (off == 0 && count == r->size))) {
		r->blocks[ifirst].dev_valid = 0;
		r->blocks[ifirst].swp_valid = 1;
	}
	else if (ifirst < ilast) {
		if (off % BLOCKSIZE == 0) {	// first block
			r->blocks[ifirst].dev_valid = 0;
			r->blocks[ifirst].swp_valid = 1;
		}
		if (end % BLOCKSIZE == 0 || end == r->size) {	// last block
			r->blocks[ilast].dev_valid = 0;
			r->blocks[ilast].swp_valid = 1;
		}
		for (iblock = ifirst + 1; iblock < ilast; iblock++) {	// the rest
			r->blocks[iblock].dev_valid = 0;
			r->blocks[iblock].swp_valid = 1;
		}
	}

	// Then, copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted). skipped[]
	// records whether a block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_htod_block(r, off, s, size, iblock, 1,
				skipped + (iblock - ifirst));
		if (ret != 0)
			goto finish;
		s += size;
		off += size;
	}

	// Finally, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->swp_addr);
	s = (void *)src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_htod_block(r, off, s, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		s += size;
		off += size;
	}

finish:
	free(skipped);
	return ret;
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
	void *s = (void *)src;
	int ret = 0;

	gprint(DEBUG, "htod: r(%p %p %ld) dst(%p) src(%p) count(%lu)\n", \
			r, r->swp_addr, r->size, dst, src, count);

	if (r->flags & HINT_PTARRAY)
		return gmm_htod_pta(r, dst, src, count);

	off = (unsigned long)(dst - r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)calloc(ilast - ifirst + 1, sizeof(char));
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// Copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted).
	// skipped[] records whether each block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_htod_block(r, off, s, size, iblock, 1,
				skipped + (iblock - ifirst));
		if (ret != 0)
			goto finish;
		s += size;
		off += size;
	}

	// Then, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->swp_addr);
	s = (void *)src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_htod_block(r, off, s, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		s += size;
		off += size;
	}

finish:
	free(skipped);
	return ret;
}
#endif

static int gmm_memset_pta(struct region *r, void *dst, int value, size_t count)
{
	unsigned long off = (unsigned long)(dst - r->swp_addr);
	// We don't do alignment checks for memset
	memset(r->pta_addr + off, value, count);
	return 0;
}

// TODO: this is a workaround implementation.
static int gmm_memset(struct region *r, void *dst, int value, size_t count)
{
	unsigned long off, end, size;
	int ifirst, ilast, iblock;
	char *skipped;
	int ret = 0;
	void *s;

	gprint(DEBUG, "memset: r(%p %p %ld) dst(%p) value(%d) count(%lu)\n", \
			r, r->swp_addr, r->size, dst, value, count);

	if (r->flags & HINT_PTARRAY)
		return gmm_memset_pta(r, dst, value, count);

	// The temporary source buffer holding %value's
	s = malloc(BLOCKSIZE);
	if (!s) {
		gprint(FATAL, "malloc failed for memset temp buffer: %s\n", \
				strerror(errno));
		return -1;
	}
	memset(s, value, BLOCKSIZE);

	off = (unsigned long)(dst - r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		free(s);
		return -1;
	}

	// Copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely because it's being evicted).
	// skipped[] records whether each block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_htod_block(r, off, s, size, iblock, 1,
				skipped + (iblock - ifirst));
		if (ret != 0)
			goto finish;
		off += size;
	}

	// Then, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->swp_addr);
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_htod_block(r, off, s, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		off += size;
	}

finish:
	free(skipped);
	free(s);
	return ret;
}

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
	int ret = 0;

/*	GMM_DPRINT("gmm_dtoh_block: r(%p) dst(%p) off(%lu)" \
			" size(%lu) block(%d) swp_valid(%d) dev_valid(%d)\n", \
			r, dst, off, size, iblock, b->swp_valid, b->dev_valid);
*/
	if (b->swp_valid) {
		memcpy(dst, r->swp_addr + off, size);
		if (skipped)
			*skipped = 0;
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
		memcpy(dst, r->swp_addr + off, size);
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
	else { // dev_valid == 1 && swp_valid == 0
		// We don't need to pin the device memory because we are holding the
		// lock of a swp_valid=0,dev_valid=1 block, which will prevent the
		// evictor, if any, from freeing the device memory under us.
		ret = block_sync(r, iblock);
		release(&b->lock);
		if (ret != 0)
			goto finish;
		memcpy(dst, r->swp_addr + off, size);
	}

finish:
	if (skipped)
		*skipped = 0;
	return ret;
}

static int gmm_dtoh_pta(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	unsigned long off = (unsigned long)(src - r->swp_addr);

	if (off % sizeof(void *)) {
		gprint(ERROR, "offset (%lu) not aligned for pta-to-host memcpy\n", off);
		return -1;
	}
	if (count % sizeof(void *)) {
		gprint(ERROR, "count (%lu) not aligned for pta-to-host memcpy\n", count);
		return -1;
	}

	memcpy(dst, r->pta_addr + off, count);
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
	unsigned long off = (unsigned long)(src - r->swp_addr);
	unsigned long end = off + count, size;
	int ifirst = BLOCKIDX(off), iblock;
	void *d = dst;
	char *skipped;
	int ret = 0;

	gprint(DEBUG, "dtoh: r(%p %p %ld) dst(%p) src(%p) count(%lu)\n", \
			r, r->swp_addr, r->size, dst, src, count);

	if (r->flags & HINT_PTARRAY)
		return gmm_dtoh_pta(r, dst, src, count);

	skipped = (char *)malloc(BLOCKIDX(end - 1) - ifirst + 1);
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// First, copy blocks whose swp buffers contain immediate, valid data.
	iblock = ifirst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_dtoh_block(r, d, off, size, iblock, 1,
				skipped + iblock - ifirst);
		if (ret != 0)
			goto finish;
		d += size;
		off += size;
		iblock++;
	}

	// Then, copy the rest blocks.
	off = (unsigned long)(src - r->swp_addr);
	iblock = ifirst;
	d = dst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_dtoh_block(r, d, off, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		d += size;
		off += size;
		iblock++;
	}

finish:
	free(skipped);
	return ret;
}

// TODO: this is a toy implementation; data should be copied
// directly from rs to rd
static int gmm_dtod(
		struct region *rd,
		struct region *rs,
		void *dst,
		const void *src,
		size_t count)
{
	int ret = 0;
	void *temp;

	gprint(DEBUG, "dtod: rd(%p %p %ld) rs(%p %p %ld) " \
			"dst(%p) src(%p) count(%lu)\n", \
			rd, rd->swp_addr, rd->size, rs, rs->swp_addr, rs->size, \
			dst, src, count);

	temp = malloc(count);
	if (!temp) {
		gprint(FATAL, "malloc failed for temp dtod buffer: %s\n", \
				strerror(errno));
		return -1;
	}

	if (gmm_dtoh(rs, temp, src, count) < 0) {
		gprint(ERROR, "failed to copy data to temp dtod buffer\n");
		free(temp);
		return -1;
	}

	if (gmm_htod(rd, dst, temp, count) < 0) {
		gprint(ERROR, "failed to copy data from temp dtod buffer\n");
		free(temp);
		return -1;
	}

	free(temp);
	return ret;
}

// Look up a memory object by the ptr passed from user program.
// ptr should fall within the virtual memory area of the host swap buffer of
// the memory object, if it can be found.
struct region *region_lookup(struct gmm_context *ctx, const void *ptr)
{
	struct region *r = NULL;
	struct list_head *pos;
	int found = 0;

	//GMM_DPRINT("region_lookup begin: %p\n", ptr);

	acquire(&ctx->lock_alloced);
	list_for_each(pos, &(ctx->list_alloced)) {
		r = list_entry(pos, struct region, entry_alloced);
		//GMM_DPRINT("region_lookup: %p %ld\n", r->swp_addr, r->size);
		if ((unsigned long)ptr >= (unsigned long)(r->swp_addr) &&
			(unsigned long)ptr <
			((unsigned long)(r->swp_addr) + (unsigned long)(r->size))) {
			found = 1;
			break;
		}
	}
	release(&ctx->lock_alloced);

	if (!found)
		r = NULL;

	return r;
}

// Select victims for %size_needed bytes of free device memory space.
// %excls[0:%nexcl) are local regions that should not be selected.
// Put selected victims in the list %victims.
int victim_select(
		long size_needed,
		struct region **excls,
		int nexcl,
		int local_only,
		struct list_head *victims)
{
	int ret = 0;

	//GMM_DPRINT("selecting victim: size(%ld) local(%d)\n", size_needed, local_only);

#if defined(GMM_REPLACEMENT_LRU)
	ret = victim_select_lru(size_needed, excls, nexcl, local_only, victims);
#elif defined(GMM_REPLACEMENT_LFU)
	ret = victim_select_lfu(size_needed, excls, nexcl, local_only, victims);
#else
	panic("replacement policy not specified");
	ret = -1;
#endif

	return ret;
}

// NOTE: When a local region is evicted, no other parties are
// supposed to be accessing the region at the same time.
// This is not true if multiple loadings happen simultaneously,
// but this region has been locked in region_load() anyway.
// A dptr array region's data never needs to be transferred back
// from device to host because swp_valid=0,dev_valid=1 will never
// happen.
int region_evict(struct region *r)
{
	int nblocks = NRBLOCKS(r->size);
	char *skipped;
	int i, ret = 0;

	gprint(INFO, "evicting region %p\n", r);
	//gmm_print_region(r);

	if (!r->dev_addr)
		panic("dev_addr is null");
	if (region_pinned(r))
		panic("evicting a pinned region");

	skipped = (char *)calloc(nblocks, sizeof(char));
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// First round
	for (i = 0; i < nblocks; i++) {
		if (r->state == STATE_FREEING)
			goto success;
		if (try_acquire(&r->blocks[i].lock)) {
			if (!r->blocks[i].swp_valid)
				ret = block_sync(r, i);
			release(&r->blocks[i].lock);
			if (ret != 0)
				goto finish;	// this is problematic if r is freeing
			skipped[i] = 0;
		}
		else
			skipped[i] = 1;
	}

	// Second round
	for (i = 0; i < nblocks; i++) {
		if (r->state == STATE_FREEING)
			goto success;
		if (skipped[i]) {
			acquire(&r->blocks[i].lock);
			if (!r->blocks[i].swp_valid)
				ret = block_sync(r, i);
			release(&r->blocks[i].lock);
			if (ret != 0)
				goto finish;	// this is problematic if r is freeing
		}
	}

success:
	list_attached_del(pcontext, r);
	if (r->dev_addr) {
		nv_cudaFree(r->dev_addr);
		r->dev_addr = NULL;
	}
	latomic_sub(&pcontext->size_attached, r->size);
	update_attached(-r->size);
	update_detachable(-r->size);
	region_inval(r, 0);
	acquire(&r->lock);
	if (r->state == STATE_FREEING) {
		if (r->swp_addr) {
			free(r->swp_addr);
			r->swp_addr = NULL;
		}
		r->state = STATE_ZOMBIE;
	}
	else
		r->state = STATE_DETACHED;
	release(&r->lock);

	gprint(INFO, "region evicted\n");
	//gmm_print_region(r);

finish:
	free(skipped);
	return ret;
}

// NOTE: Client %client should have been pinned when this function
// is called.
int remote_victim_evict(int client, long size_needed)
{
	int ret;
	gprint(DEBUG, "remote eviction in client %d\n", client);
	ret = msq_send_req_evict(client, size_needed, 1);
	gprint(DEBUG, "remote eviction returned: %d\n", ret);
	client_unpin(client);
	return ret;
}

// Similar to gmm_evict, but only select at most one victim from local
// region list, even if it is smaller than required, evict it, and return.
int local_victim_evict(long size_needed)
{
	struct list_head victims;
	struct victim *v;
	struct region *r;
	int ret;

	gprint(DEBUG, "local eviction: %ld bytes\n", size_needed);
	INIT_LIST_HEAD(&victims);

	ret = victim_select(size_needed, NULL, 0, 1, &victims);
	if (ret != 0)
		return ret;

	if (list_empty(&victims))
		return 0;

	v = list_entry(victims.next, struct victim, entry);
	r = v->r;
	free(v);
	return region_evict(r);
}

// Evict the victim %victim.
// %victim may point to a local region or a remote client that
// may own some evictable region.
int victim_evict(struct victim *victim, long size_needed)
{
	if (victim->r)
		return region_evict(victim->r);
	else if (victim->client != -1)
		return remote_victim_evict(victim->client, size_needed);
	else {
		panic("victim is neither local nor remote");
		return -1;
	}
}

// Evict some device memory so that the size of free space can
// satisfy %size_needed. Regions in %excls[0:%nexcl) should not
// be selected for eviction.
static int gmm_evict(long size_needed, struct region **excls, int nexcl)
{
	struct list_head victims, *e;
	struct victim *v;
	int ret = 0;

	gprint(DEBUG, "evicting for %ld bytes\n", size_needed);
	INIT_LIST_HEAD(&victims);

	do {
		ret = victim_select(size_needed, excls, nexcl, 0, &victims);
		if (ret != 0)
			return ret;

		for (e = victims.next; e != (&victims); ) {
			v = list_entry(e, struct victim, entry);
			if (memsize_free() < size_needed) {
				if ((ret = victim_evict(v, size_needed)) != 0)
					goto fail_evict;
			}
			else if (v->r) {
				acquire(&v->r->lock);
				if (v->r->state != STATE_FREEING)
					v->r->state = STATE_ATTACHED;
				release(&v->r->lock);
			}
			else if (v->client != -1)
				client_unpin(v->client);
			list_del(e);
			e = e->next;
			free(v);
		}
	} while (memsize_free() < size_needed);

	gprint(DEBUG, "eviction finished\n");
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
		else if (v->client != -1)
			client_unpin(v->client);
		list_del(e);
		e = e->next;
		free(v);
	}

	gprint(DEBUG, "eviction failed\n");
	return ret;
}

// Allocate device memory to a region (i.e., attach).
static int region_attach(
		struct region *r,
		int pin,
		struct region **excls,
		int nexcl)
{
	cudaError_t cret;
	int ret;

	gprint(DEBUG, "attaching region %p\n", r);

	if (r->state != STATE_DETACHED) {
		gprint(ERROR, "nothing to attach\n");
		return -1;
	}

	// Attach if current free memory space is larger than region size.
	if (r->size <= memsize_free()) {
		if ((cret = nv_cudaMalloc(&r->dev_addr, r->size)) == cudaSuccess)
			goto attach_success;
		else {
			gprint(DEBUG, "nv_cudaMalloc failed: %s (%d)\n", \
					cudaGetErrorString(cret), cret);
			if (cret == cudaErrorLaunchFailure)
				return -1;
		}
	}

	// Evict some device memory.
	ret = gmm_evict(r->size, excls, nexcl);
	if (ret < 0 || (ret > 0 && memsize_free() < r->size))
		return ret;

	// Try to attach again.
	if ((cret = nv_cudaMalloc(&r->dev_addr, r->size)) != cudaSuccess) {
		r->dev_addr = NULL;
		gprint(DEBUG, "nv_cudaMalloc failed: %s (%d)\n", \
				cudaGetErrorString(cret), cret);
		if (cret == cudaErrorLaunchFailure)
			return -1;
		else
			return 1;
	}

attach_success:
	latomic_add(&pcontext->size_attached, r->size);
	update_attached(r->size);
	update_detachable(r->size);
	if (pin)
		region_pin(r);
	// Reassure that the dev copies of all blocks are set to invalid.
	region_inval(r, 0);
	r->state = STATE_ATTACHED;
	list_attached_add(pcontext, r);

	gprint(DEBUG, "region attached\n");
	return 0;
}

// Load a region to device memory.
// excls[0:nexcl) are regions that should not be evicted
// during the loading.
static int region_load(
		struct region *r,
		int pin,
		struct region **excls,
		int nexcl)
{
	int i, ret = 0;

	gprint(DEBUG, "loading region r(%p %p %lu %d %d)\n", \
			r, r->swp_addr, r->size, r->flags, r->state);
	//gmm_print_region(r);

	if (r->state == STATE_EVICTING) {
		gprint(ERROR, "should not see an evicting region\n");
		return -1;
	}

	// Attach if the region is still detached.
	if (r->state == STATE_DETACHED) {
		if ((ret = region_attach(r, 1, excls, nexcl)) != 0)
			return ret;
	}
	else {
		if (pin)
			region_pin(r);
		// Update the region's position in the LRU list.
		list_attached_mov(pcontext, r);
	}

	// Fetch data to device memory if necessary. A dptr array region will
	// be synced later after all other referenced regions have been loaded.
	if ((r->rwhint.flags & HINT_READ) && !(r->flags & HINT_PTARRAY)) {
		for (i = 0; i < NRBLOCKS(r->size); i++) {
			acquire(&r->blocks[i].lock);	// Though this is useless
			if (!r->blocks[i].dev_valid)
				ret = block_sync(r, i);
			release(&r->blocks[i].lock);
			if (ret != 0)
				return ret;
		}
	}

	gprint(DEBUG, "loaded region\n");
	return 0;
}

// Load all %n regions specified by %rgns to device.
// Every successfully loaded region is pinned to device.
// If all regions cannot be loaded successfully, successfully
// loaded regions will be unpinned so that they can be
// replaced by other kernel launches.
// Return value: 0 - success; < 0 - fatal failure; > 0 - retry later.
static int gmm_load(struct region **rgns, int n)
{
	char *pinned;
	int i, ret;

	if (n == 0)
		return 0;
	if (n < 0 || (n > 0 && !rgns))
		return -1;

	gprint(DEBUG, "gmm_load begins: %d\n", n);

	pinned = (char *)calloc(n, sizeof(char));
	if (!pinned) {
		gprint(FATAL, "malloc failed for pinned array: %s\n", strerror(errno));
		return -1;
	}

	for (i = 0; i < n; i++) {
		if (rgns[i]->state == STATE_FREEING || rgns[i]->state == STATE_ZOMBIE) {
			gprint(ERROR, "cannot load a freed region r(%p %p %ld %d %d)\n", \
					rgns[i], rgns[i]->swp_addr, rgns[i]->size, \
					rgns[i]->flags, rgns[i]->state);
			ret = -1;
			goto fail;
		}
		// NOTE: In current design, this locking is redundant
		acquire(&rgns[i]->lock);
		ret = region_load(rgns[i], 1, rgns, n);
		release(&rgns[i]->lock);
		if (ret != 0)
			goto fail;
		pinned[i] = 1;
	}

	gprint(DEBUG, "gmm_load finished\n");
	free(pinned);
	return 0;

fail:
	for (i = 0; i < n; i++)
		if (pinned[i])
			region_unpin(rgns[i]);
	free(pinned);
	gprint(DEBUG, "gmm_load failed\n");
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
	if (status != cudaSuccess)
		gprint(ERROR, "kernel failed: %d\n", status);
	else
		gprint(DEBUG, "kernel succeeded\n");
	for (i = 0; i < pcb->nrgns; i++) {
		if (pcb->flags[i] & HINT_WRITE)
			atomic_dec(&pcb->rgns[i]->writing);
		if (pcb->flags[i] & HINT_READ)
			atomic_dec(&pcb->rgns[i]->reading);
		region_unpin(pcb->rgns[i]);
	}
	free(pcb);
}

// Here we utilize CUDA 5+'s stream callback feature to capture kernel
// finish event and unpin related regions accordingly.
static int gmm_launch(const char *entry, struct region **rgns, int nrgns)
{
	cudaError_t cret;
	struct kcb *pcb;
	int i;

	if (nrgns > NREFS) {
		gprint(ERROR, "too many regions\n");
		return -1;
	}

	pcb = (struct kcb *)malloc(sizeof(*pcb));
	if (!pcb) {
		gprint(FATAL, "malloc failed for kcb: %s\n", strerror(errno));
		return -1;
	}
	if (nrgns > 0)
		memcpy(pcb->rgns, rgns, sizeof(void *) * nrgns);
	for (i = 0; i < nrgns; i++) {
		pcb->flags[i] = rgns[i]->rwhint.flags;
		if (pcb->flags[i] & HINT_WRITE)
			atomic_inc(&rgns[i]->writing);
		if (pcb->flags[i] & HINT_READ)
			atomic_inc(&rgns[i]->reading);
	}
	pcb->nrgns = nrgns;

	if ((cret = nv_cudaLaunch(entry)) != cudaSuccess) {
		for (i = 0; i < nrgns; i++) {
			if (pcb->flags[i] & HINT_WRITE)
				atomic_dec(&pcb->rgns[i]->writing);
			if (pcb->flags[i] & HINT_READ)
				atomic_dec(&pcb->rgns[i]->reading);
		}
		free(pcb);
		gprint(ERROR, "nv_cudaLaunch failed: %s (%d)\n", \
				cudaGetErrorString(cret), cret);
		return -1;
	}
	nv_cudaStreamAddCallback(stream_issue, gmm_kernel_callback, (void *)pcb, 0);

	// TODO: update this client's position in global LRU client list

	gprint(DEBUG, "kernel launched\n");
	return 0;
}
