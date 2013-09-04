#ifndef _GMM_ATOMIC_H_
#define _GMM_ATOMIC_H_

// Integers
static inline int atomic_add(int *ptr, int val)
{
	return __sync_fetch_and_add(ptr, val);
}

static inline int atomic_inc(int *ptr)
{
	return __sync_fetch_and_add(ptr, 1);
}

static inline int atomic_sub(int *ptr, int val)
{
	return __sync_fetch_and_sub(ptr, val);
}

static inline int atomic_dec(int *ptr)
{
	return __sync_fetch_and_sub(ptr, 1);
}

// Long integers
static inline long atomic_addl(long *ptr, long val)
{
	return __sync_fetch_and_add(ptr, val);
}

static inline long atomic_incl(long *ptr)
{
	return __sync_fetch_and_add(ptr, 1);
}

static inline long atomic_subl(long *ptr, long val)
{
	return __sync_fetch_and_sub(ptr, val);
}

static inline long atomic_decl(long *ptr)
{
	return __sync_fetch_and_sub(ptr, 1);
}

#endif
