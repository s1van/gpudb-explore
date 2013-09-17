#include "util.h"

void list_alloced_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_add(&r->entry_alloced, &ctx->list_alloced);
	release(&ctx->lock_alloced);
}

void list_alloced_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_del(&r->entry_alloced);
	release(&ctx->lock_alloced);
}

void list_attached_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_add(&r->entry_attached, &ctx->list_attached);
	release(&ctx->lock_attached);
}

void list_attached_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_del(&r->entry_attached);
	release(&ctx->lock_attached);
}

// Look up a memory object by the ptr passed from user program.
// ptr should fall within the virtual memory area of the host swap buffer of
// the memory object, if it can be found.
struct region *region_lookup(struct gmm_context *ctx, const void *ptr)
{
	struct region *r;
	struct list_head *pos;
	int found = 0;

	acquire(&ctx->lock_alloced);
	list_for_each(pos, &ctx->list_alloced) {
		r = list_entry(pos, struct region, entry_alloced);
		if ((unsigned long)ptr >= (unsigned long)r->addr_swp &&
			(unsigned long)ptr < ((unsigned long)r->addr_swp + r->size)) {
			found = 1;
			break;
		}
	}
	release(&ctx->lock_alloced);

	if (!found)
		r = NULL;

	return r;
}
