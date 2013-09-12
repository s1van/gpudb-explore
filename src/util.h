#ifndef _GMM_UTIL_H_
#define _GMM_UTIL_H_

#include "core.h"

#define MIN(x, y)	((x) < (y) ? (x) : (y))

void list_alloced_add(struct gmm_context *ctx, struct region *r);
void list_alloced_del(struct gmm_context *ctx, struct region *r);
void list_attached_add(struct gmm_context *ctx, struct region *r);
void list_attached_del(struct gmm_context *ctx, struct region *r);
struct region *region_lookup(struct gmm_context *ctx, const void *ptr);

// Whether pointer p is included in pointer array a[0:n)
static inline int is_included(void **a, int n, void *p)
{
	int i;

	for (i = 0; i < n; i++)
		if (a[i] == p)
			return 1;

	return 0;
}

#endif
