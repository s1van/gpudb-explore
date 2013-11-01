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

// For internal use only
void gmm_print_region(void *rgn)
{
	struct region *r = (struct region *)rgn;

	gprint(DEBUG, "printing dptr %p (%p)\n", r->swp_addr, r);
	gprint(DEBUG, "\tsize: %ld\t\tstate: %d\t\tflags: %d\n", \
			r->size, r->state, r->flags);
	gprint(DEBUG, "\tdev_addr: %p\t\tswp_addr: %p\t\tpta_addr: %p\n", \
			r->dev_addr, r->swp_addr, r->pta_addr);
	gprint(DEBUG, "\tpinned: %d\t\twriting: %d\t\treading: %d\n", \
			atomic_read(&r->pinned), atomic_read(&r->writing), \
			atomic_read(&r->reading));
}

// Print info of the region containing %dptr
GMM_EXPORT
void gmm_print_dptr(const void *dptr)
{
	struct region *r;

	r = region_lookup(pcontext, dptr);
	if (!r) {
		gprint(DEBUG, "failed to look up region containing %p\n", dptr);
		return;
	}
	gmm_print_region(r);
}
