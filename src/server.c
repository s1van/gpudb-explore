// Set up the global shared memory

#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */
#include <semaphore.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <cuda.h>

#include "gmm_protocol.h"

int start_gmm()
{
	cudaError_t ret;
	size_t free, total;
	sem_t sem;
	int i, shmfd;
	struct gmm_global *pglobal;

	ret = cudaMemGetInfo(&free, &total);
	if (ret != cudaSuccess) {
		fprintf(stderr, "Failed to get CUDA device memory info: %s\n",
				cudaGetErrorString(ret));
		exit(-1);
	}

	fprintf(stderr, "Total size of GPU memory: %u bytes.\n", total);
	fprintf(stderr, "Size of free GPU memory: %u bytes.\n", free);
	fprintf(stderr, "Setting GMM global state...\n");

	// Create the semaphore for syncing access to shared memory
	sem = sem_open(GMM_SEM_NAME, O_CREAT | O_EXCL, 0666, 1);
	if (sem == SEM_FAILED && errno == EEXIST) {
		perror("Semaphore already exists; have to unlink it and link again");
		if (sem_unlink(GMM_SEM_NAME) == -1) {
			perror("Failed to remove semaphore; quitting");
			exit(1);
		}
		sem = sem_open(GMM_SEM_NAME, O_CREAT | O_EXCL, 0666, 1);
		if (sem == SEM_FAILED) {
			perror("Failed to create semaphore again; quitting");
			exit(1);
		}
	}
	else if (sem == SEM_FAILED) {
		perror("Failed to create semaphore; exiting");
		exit(1);
	}
	// The semaphore link has been created; we can close it now
	sem_close(sem);

	// Create, truncate, and initialize the shared memory
	shmfd = shm_open(GMM_SHM_NAME, O_RDWR | O_CREAT | O_EXCL, 0644);
	if (shmfd == -1 && errno == EEXIST) {
		perror("Shared memory already exists; unlink it and link again");
		if (shm_unlink(GMM_SHM_NAME) == -1) {
			perror("Failed to remove shared memory; quitting");
			goto fail_shm;
		}
		shmfd = shm_open(GMM_SHM_NAME, O_RDWR | O_CREAT | O_EXCL, 0644);
		if (shmfd == -1) {
			perror("Failed to create shared memory again; quitting");
			goto fail_shm;
		}
	}
	else if (shmfd == -1) {
		perror("Failed to create shared memory; quitting");
		goto fail_shm;
	}

	if (ftruncate(shmfd, sizeof(struct gmm_global)) == -1) {
		perror("Failed to truncate the shared memory; quitting");
		goto fail_truncate;
	}

	pglobal = (struct gmm_global *)mmap(NULL, sizeof(*pglobal),
			PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
	if (pglobal == MAP_FAILED) {
		perror("failed to mmap the shared memory; quitting");
		goto fail_mmap;
	}

	pglobal->mem_total = total;
	pglobal->mem_used = 0;
	pglobal->maxclients = NCLIENTS;
	pglobal->nclients = 0;
	pglobal->ilru = pglobal->imru = -1;
	memset(pglobal->clients, 0, sizeof(pglobal->clients[0]) * NCLIENTS);
	for (i = 0; i < NCLIENTS; i++) {
		pglobal->clients[i].index = -1;
		pglobal->clients[i].list_client.next = -1;
		pglobal->clients[i].list_client.prev = -1;
	}

	// Now we can unmap and close the shared memory; then we're done!
	munmap(pglobal, sizeof(*pglobal));	// or maybe we can skip unmapping
	close(shmfd);

	fprintf(stderr, "Setting done!\nPlease restart all processes using GMM.\n");
	return 0;

fail_mmap:
fail_truncate:
	close(shmfd);
	shm_unlink(GMM_SHM_NAME);
fail_shm:
	sem_unlink(GMM_SEM_NAME);

	return -1;
}

int stop_gmm()
{
	if (shm_unlink(GMM_SHM_NAME) == -1) {
		perror("Failed to unlink shared memory");
	}

	if (sem_unlink(GMM_SEM_NAME) == -1) {
		perror("Failed to unlink semaphore");
	}

	return 0;
}

int main(int argc, char *argv[])
{
	return start_gmm();
}
