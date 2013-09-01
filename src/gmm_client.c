#include "gmm_protocol.h"


int gmm_reclaim()
{
	if (shm_mc_id != 0)
		shmctl(shm_mc_id, IPC_RMID, 0);
	return 0;
}

int gmm_attach()
{
	//init arg structure for asynchorous functions
	targs = NEW_GMM_ARGS();

	// Get the semaphore
	mutex = sem_open(GMM_SEM_NAME, 0);
	if (mutex == SEM_FAILED) {
		perror("Unable to get semaphore");
		return -1;
	}

	//get the segment id
	if ((shm_mc_id = shmget((key_t)GMM_SHARED, GMM_SHARED_SIZE, 0666)) < 0) {
		perror("shmget");
		exit(1);
	}

	//attach the shared segment
	if ((gmm_sdata = shmat(shm_mc_id, NULL, 0)) == (gmm_shared) -1) {
		perror("shmat");
		exit(1);
	}

	init_gmm_local(&gmm_pdata);

	sem_wait(mutex);
	gmm_id = new_gmm_id(gmm_sdata);
	S_INC_PNUM(gmm_sdata);
	sem_post(mutex);

	//request for new stream for current process
	if(!mystream)
		nv_cudaStreamCreate(&mystream);

	return 0;
}

int gmm_detach(){
	sem_wait(mutex);
	clean_gmm_shared(gmm_sdata, gmm_id);
	S_DEC_PNUM(gmm_sdata);
	sem_post(mutex);

	sem_close(mutex);
	sem_unlink(GMM_SEM_NAME);
	destroy_gmm_local(gmm_pdata);
	return 0;
}
