// The interfaces exported to allow user programs to print GMM runtime
// info for debugging purposes.
// The user program should first open libgmm.so with RTLD_NOLOAD flag
// to get the handle to the already loaded shared library. Then use
// dlsym to get the addresses of related interfaces.
#include "common.h"
#include "atomic.h"
#include "core.h"

struct region *region_lookup(struct gmm_context *ctx, const void *ptr);

extern struct gmm_context *pcontext;

// Print info of the region containing %dptr
void gmm_print_dptr(const void *dptr)
{
#ifdef GMM_DEBUG
	struct region *r;

	r = region_lookup(pcontext, dptr);
	if (!r) {
		GMM_DPRINT("failed to look up region containing %p\n", dptr);
		return;
	}

	GMM_DPRINT("printing dptr %p (%p)\n", dptr, r);
	GMM_DPRINT("\tsize: %ld\t\tstate: %d\n", r->size, r->state);
	GMM_DPRINT("\tdev_addr: %p\t\tswp_addr: %p\n", r->dev_addr, r->swp_addr);
	GMM_DPRINT("\tpinned: %d\t\trwhint: %x\n", atomic_read(&r->pinned), \
			(unsigned int)(r->rwhint.flags));
#endif
}
