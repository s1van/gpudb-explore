#include <errno.h>
#include "replacement.h"
#include "protocol.h"
#include "util.h"

extern struct gmm_context *pcontext;


// Select the LRU region in local context.
int victim_select_lru_local(
		long size_needed,
		struct region **excls,
		int nexcl,
		struct list_head *victims)
{
	struct list_head *pos;
	struct victim *v;
	struct region *r;

	v = (struct victim *)malloc(sizeof(*v));
	if (!v) {
		GMM_DPRINT("malloc failed for a new victim: %s\n", strerr(errno));
		return -1;
	}
	v->client = -1;

	acquire(&pcontext->lock_attached);
	list_for_each_prev(pos, &pcontext->list_attached) {
		r = list_entry(pos, struct region, entry_attached);
		if (!is_included(excls, nexcl, r) && try_acquire(&r->lock)) {
			if (r->state == STATE_ATTACHED && !region_pinned(r)) {
				r->state = STATE_EVICTING;
				release(&r->lock);
				v->r = r;
				list_add(&v->entry, victims);
				break;
			}
			else
				release(&r->lock);
		}
	}
	release(&pcontext->lock_attached);

	if (pos == &pcontext->list_attached) {
		free(v);
		if (size_needed > free_memsize())
			return -1;
	}

	return 0;
}

// In the least recently used client, select the least recently
// used detachable region as the victim.
int victim_select_lru(
		long size_needed,
		struct region **excls,
		int nexcl,
		struct list_head *victims)
{
	struct victim *v;
	int iclient;

	iclient = client_lru_detachable();
	if (iclient == -1)
		return -1;

	// If the LRU client is a remote client, record its client id;
	// otherwise, select the LRU region in the local context immediately.
	if (!is_client_local(iclient)) {
		v = (struct victim *)malloc(*v);
		if (!v) {
			GMM_DPRINT("malloc failed for a new victim: %s\n", strerr(errno));
			client_unpin(iclient);
			return -1;
		}
		v->r = NULL;
		v->client = iclient;
		list_add(&v->entry, victims);
		return 0;
	}
	else
		return victim_select_lru_local(size_needed, excls, nexcl, victims);
}

// TODO
int victim_select_lfu(
		long size_needed,
		struct region **excls,
		int nexcl,
		struct list_head *victims)
{
	return -1;
}
