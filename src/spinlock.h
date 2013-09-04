#ifndef _GMM_SPINLOCK_H_
#define _GMM_SPINLOCK_H_

struct spinlock {
	unsigned int locked;
};

static inline unsigned int xchg(
		volatile unsigned int *addr,
		unsigned int newval)
{
  unsigned int result;

  // The + in "+m" denotes a read-modify-write operand.
  asm volatile("lock; xchgl %0, %1" :
               "+m" (*addr), "=a" (result) :
               "1" (newval) :
               "cc");
  return result;
}

void initlock(struct spinlock *lk);
void acquire(struct spinlock *lk);
int try_acquire(struct spinlock *lk);
void release(struct spinlock *lk);

#endif
