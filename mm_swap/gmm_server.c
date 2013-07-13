#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <semaphore.h>

#include "./gmm_meta_interface.h"
#include "./gmm_meta.h"

int main()
{
	char x;
	gmm_init_attach(GPU_MEM_SIZE);
	printf("GPU Memory Available: %ld\n", gmm_getFreeMem());
	while(1) {
		x=getchar();
		if (x == 'r') {
			gmm_setFreeMem(GPU_MEM_SIZE);
			printf("GPU Memory Available: %ld\n", gmm_getFreeMem());
		} else if (x == 'q') break;
	}
	gmm_detach();
	gmm_reclaim();	

	return 0;
}

