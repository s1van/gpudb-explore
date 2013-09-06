#include "util.h"

void list_alloced_add(struct gmm_context *ctx, struct region *m)
{
	acquire(&ctx->lock);
	list_add(&m->entry_alloced, &ctx->list_alloced);
	release(&ctx->lock);
}

void list_alloced_del(struct gmm_context *ctx, struct region *m)
{
	acquire(&ctx->lock);
	list_del(&m->entry_alloced);
	release(&ctx->lock);
}

// Look up a memory object by the ptr passed from user program.
// ptr should fall within the virtual memory area of the host swap buffer of
// the memory object, if it can be found.
struct region *region_lookup(struct gmm_context *ctx, void *ptr)
{
	struct region *r;
	struct list_head *pos;
	int found = 0;

	acquire(&ctx->lock);
	list_for_each(pos, &ctx->list_alloced) {
		r = list_entry(pos, struct region, entry_alloced);
		if ((unsigned long)ptr >= (unsigned long)r->addr_swp &&
			(unsigned long)ptr < ((unsigned long)r->addr_swp + r->size)) {
			found = 1;
			break;
		}
	}
	release(&ctx->lock);

	if (!found)
		r = NULL;

	return r;
}
