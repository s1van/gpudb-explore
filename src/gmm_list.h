#ifndef _GMM_LIST_H_
#define _GMM_LIST_H_

// Here is a copy of Linux's implementation of double linked list and the
// related operations that are interesting to us.

struct list_head {
	struct list_head *prev;
	struct list_head *next;
};

static inline void INIT_LIST_HEAD(struct list_head *list)
{
	list->next = list;
	list->prev = list;
}

// Insert a new entry between two known consecutive entries.
static inline void __list_add(
		struct list_head *add,
		struct list_head *prev,
		struct list_head *next)
{
	next->prev = add;
	add->next = next;
	add->prev = prev;
	prev->next = add;
}

// Add a new entry.
static inline void list_add(struct list_head *add, struct list_head *head)
{
	__list_add(add, head, head->next);
}

// Add a new entry to the tail.
static inline void list_add_tail(struct list_head *add, struct list_head *head)
{
	__list_add(add, head->prev, head);
}

// Delete a list entry by making the prev/next entries point to each other.
static inline void __list_del(struct list_head * prev, struct list_head * next)
{
	next->prev = prev;
	prev->next = next;
}

// Delete an entry from list.
static inline void list_del(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
	entry->next = NULL;
	entry->prev = NULL;
}

#define offsetof(TYPE, MEMBER) ((unsigned long) &((TYPE *)0)->MEMBER)

// Get the struct containing this list_end entry.
#define list_entry(ptr, type, member) \
	((type *) ((char *)(ptr) - offsetof(type,member)))

#endif
