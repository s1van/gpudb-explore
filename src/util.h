#ifndef _GMM_UTIL_H_
#define _GMM_UTIL_H_

#include "core.h"

#define MIN(x, y)	((x) < (y) ? (x) : (y))

void list_alloced_add(struct gmm_context *ctx, struct region *m);
void list_alloced_del(struct gmm_context *ctx, struct region *m);
struct region *region_lookup(struct gmm_context *ctx, void *ptr);

#endif
