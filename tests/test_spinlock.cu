// spinlock
#include <cuda.h>

#include "test.h"
#include "gmm.h"
#include "spinlock.h"

int test_spinlock()
{
	struct spinlock lock;

	GMM_TPRINT("lock addr: %p\n", &lock);
	initlock(&lock);
	GMM_TPRINT("before lock: %u\n", lock.locked);
	acquire(&lock);
	GMM_TPRINT("within lock: %u\n", lock.locked);
	release(&lock);
	GMM_TPRINT("after lock: %u\n", lock.locked);

	return 0;
}
