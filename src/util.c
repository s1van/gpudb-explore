#include "util.h"

void list_alloced_add(struct gmm_local *l, struct memobj *m)
{
	acquire(&l->lock);
	list_add(&m->entry_alloced, &l->list_alloced);
	release(&l->lock);
}

void list_alloced_del(struct gmm_local *l, struct memobj *m)
{
	acquire(&l->lock);
	list_del(&m->entry_alloced);
	release(&l->lock);
}
