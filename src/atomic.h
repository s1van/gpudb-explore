#ifndef _GMM_ATOMIC_H_
#define _GMM_ATOMIC_H_

typedef int atomic_t;
typedef long atomic_l_t;

// Integers
static inline void atomic_set(atomic_t *ptr, int val)
{
	*ptr = val;
}

static inline int atomic_read(atomic_t *ptr)
{
	return *ptr;
}

static inline int atomic_add(atomic_t *ptr, int val)
{
	return __sync_fetch_and_add(ptr, val);
}

static inline int atomic_inc(atomic_t *ptr)
{
	return __sync_fetch_and_add(ptr, 1);
}

static inline int atomic_sub(atomic_t *ptr, int val)
{
	return __sync_fetch_and_sub(ptr, val);
}

static inline int atomic_dec(atomic_t *ptr)
{
	return __sync_fetch_and_sub(ptr, 1);
}

// Long integers
static inline void atomic_setl(atomic_l_t *ptr, int val)
{
	*ptr = val;
}

static inline int atomic_readl(atomic_l_t *ptr)
{
	return *ptr;
}

static inline long atomic_addl(atomic_l_t *ptr, long val)
{
	return __sync_fetch_and_add(ptr, val);
}

static inline long atomic_incl(atomic_l_t *ptr)
{
	return __sync_fetch_and_add(ptr, 1);
}

static inline long atomic_subl(atomic_l_t *ptr, long val)
{
	return __sync_fetch_and_sub(ptr, val);
}

static inline long atomic_decl(atomic_l_t *ptr)
{
	return __sync_fetch_and_sub(ptr, 1);
}

#endif
