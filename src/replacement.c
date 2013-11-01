#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "replacement.h"
#include "protocol.h"
#include "common.h"
#include "client.h"

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

	//GMM_DPRINT("victim_select_lru_local: size(%ld) victims(%p) prev(%p) next(%p)\n", size_needed, victims, victims->prev, victims->next);

	v = (struct victim *)malloc(sizeof(*v));
	if (!v) {
		gprint(FATAL, "malloc failed for a new victim: %s\n", strerror(errno));
		return -1;
	}
	v->client = -1;

	acquire(&pcontext->lock_attached);
	list_for_each_prev(pos, &pcontext->list_attached) {
		r = list_entry(pos, struct region, entry_attached);
		if (!is_included((void **)excls, nexcl, (void *)r) &&
				try_acquire(&r->lock)) {
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
		gprint(DEBUG, "no local victim selected\n");
		free(v);
		return 1;	// XXX: this can be problematic
	}
	gprint(DEBUG, "victim selected r(%p %p %ld %d %d)\n", \
			r, r->swp_addr, r->size, r->flags, r->state);

	return 0;
}

// In the least recently used client, select the least recently
// used detachable region as the victim.
int victim_select_lru(
		long size_needed,
		struct region **excls,
		int nexcl,
		int local_only,
		struct list_head *victims)
{
	struct victim *v;
	int iclient;

	gprint(DEBUG, "selecting victim for %ld bytes\n", size_needed);

	if (!local_only) {
		iclient = client_lru_detachable();
		if (iclient == -1)
			return 1;
	}

	// If the LRU client is a remote client, record its client id;
	// otherwise, select the LRU region in the local context immediately.
	if (!local_only && !is_client_local(iclient)) {
		v = (struct victim *)malloc(sizeof(*v));
		if (!v) {
			gprint(FATAL, "malloc failed for a new victim: %s\n", \
					strerror(errno));
			client_unpin(iclient);
			return -1;
		}
		v->r = NULL;
		v->client = iclient;
		list_add(&v->entry, victims);
		gprint(DEBUG, "remote victim selected from client %d\n", iclient);
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
		int local_only,
		struct list_head *victims)
{
	return -1;
}
