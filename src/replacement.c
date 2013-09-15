#include "replacement.h"
#include "protocol.h"

extern struct gmm_global *pglobal;

// The LRU region is the LRU region in the LRU client.
int victim_select_lru(
		long size_needed,
		struct region **excls,
		int nexcl,
		struct list_head *victims)
{
	int iclient;

	for (iclient = pglobal->ilru)
	return 0;
}

int victim_select_lfu(
		long size_needed,
		struct region **excls,
		int nexcl,
		struct list_head *victims)
{
	return 0;
}
