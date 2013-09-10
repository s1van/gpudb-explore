#ifndef _GMM_UTIL_H_
#define _GMM_UTIL_H_

#include "core.h"

#define MIN(x, y)	((x) < (y) ? (x) : (y))

void list_alloced_add(struct gmm_context *ctx, struct region *r);
void list_alloced_del(struct gmm_context *ctx, struct region *r);
void list_attached_add(struct gmm_context *ctx, struct region *r);
void list_attached_del(struct gmm_context *ctx, struct region *r);
struct region *region_lookup(struct gmm_context *ctx, void *ptr);

#define region_pinned(r)	atomic_read(&r->pinned)

#endif
