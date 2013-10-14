// Set up the global GMM environment

#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */
#include <getopt.h>
#include <semaphore.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "protocol.h"

int start(size_t mem_total)
{
	struct gmm_global *pglobal;
	cudaError_t ret;
	size_t free, total;
	sem_t *sem;
	int i, shmfd;

	ret = cudaMemGetInfo(&free, &total);
	if (ret != cudaSuccess) {
		fprintf(stderr, "Failed to get CUDA device memory info: %s\n",
				cudaGetErrorString(ret));
		exit(-1);
	}

	fprintf(stderr, "Total GPU memory: %lu bytes.\n", total);
	fprintf(stderr, "Free GPU memory: %lu bytes.\n", free);
	if (mem_total > 0 && mem_total <= total)
		total = mem_total;
	else
		total -= 200000000;		// FIXME
	fprintf(stderr, "Use GPU memory: %lu bytes.\n", total);
	fprintf(stderr, "Setting GMM ...\n");

	// Create the semaphore for syncing kernel launches.
	sem = sem_open(GMM_SEM_LAUNCH, O_CREAT | O_EXCL, 0666, 1);
	if (sem == SEM_FAILED && errno == EEXIST) {
		perror("Semaphore already exists; have to unlink it and link again");
		if (sem_unlink(GMM_SEM_LAUNCH) == -1) {
			perror("Failed to remove semaphore; quitting");
			exit(1);
		}
		sem = sem_open(GMM_SEM_LAUNCH, O_CREAT | O_EXCL, 0666, 1);
		if (sem == SEM_FAILED) {
			perror("Failed to create semaphore again; quitting");
			exit(1);
		}
	}
	else if (sem == SEM_FAILED) {
		perror("Failed to create semaphore; exiting");
		exit(1);
	}
	// The semaphore link has been created; we can close it now.
	sem_close(sem);

	// Create, truncate, and initialize the shared memory.
	// TODO: a better procedure is to disable access to shared memory first
	// (by setting permission), set up content, and then enable access.
	shmfd = shm_open(GMM_SHM_GLOBAL, O_RDWR | O_CREAT | O_EXCL, 0644);
	if (shmfd == -1 && errno == EEXIST) {
		perror("Shared memory already exists; unlink it and link again");
		if (shm_unlink(GMM_SHM_GLOBAL) == -1) {
			perror("Failed to remove shared memory; quitting");
			goto fail_shm;
		}
		shmfd = shm_open(GMM_SHM_GLOBAL, O_RDWR | O_CREAT | O_EXCL, 0644);
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
	latomic_set(&pglobal->mem_used, 0);
	initlock(&pglobal->lock);
	pglobal->nclients = 0;
	pglobal->ilru = pglobal->imru = -1;
	memset(pglobal->clients, 0, sizeof(pglobal->clients[0]) * NCLIENTS);
	for (i = 0; i < NCLIENTS; i++) {
		pglobal->clients[i].index = -1;
		pglobal->clients[i].inext = -1;
		pglobal->clients[i].iprev = -1;
	}

	munmap(pglobal, sizeof(*pglobal));
	close(shmfd);

	fprintf(stderr, "Setting done!\nGMM started.\n");
	return 0;

fail_mmap:
fail_truncate:
	close(shmfd);
	shm_unlink(GMM_SHM_GLOBAL);
fail_shm:
	sem_unlink(GMM_SEM_LAUNCH);

	return -1;
}

int stop()
{
	if (shm_unlink(GMM_SHM_GLOBAL) == -1) {
		perror("Failed to unlink shared memory");
	}

	if (sem_unlink(GMM_SEM_LAUNCH) == -1) {
		perror("Failed to unlink semaphore");
	}

	printf("GMM stopped\n");
	return 0;
}

int restart(size_t mem_total)
{
	stop();
	return start(mem_total);
}

int main(int argc, char *argv[])
{
	struct option options[5];
	char *opts = "serm:";
	int command = '\0';
	long mem_total = 0;
	int c, ret = 0;

	memset(options, 0, sizeof(options[0]) * 5);
	options[0].name = "start";
	options[0].val = 's';
	options[1].name = "stop";
	options[1].val = 'e';
	options[2].name = "restart";
	options[2].val = 'r';
	options[3].name = "mem-size";
	options[3].val = 'm';
	options[3].has_arg = 1;

	while ((c = getopt_long(argc, argv, opts, options, NULL)) != -1) {
		switch (c) {
		case 's':
			command = 's';
			break;
		case 'e':
			command = 'e';
			break;
		case 'r':
			command = 'r';
			break;
		case 'm':
			mem_total = atol(optarg);
			break;
		case '?':
			if (optopt == 'm')
				fprintf(stderr, "Please specify memory size\n");
			else
				fprintf(stderr, "Unknown option %c\n", c);
			return -1;
			break;
		default:
			abort();
			break;
		}
	}

	if (command == 's')
		ret = start(mem_total);
	else if (command == 'e')
		ret = stop();
	else if (command == 'r')
		ret =restart(mem_total);
	else {
		fprintf(stderr, "No action specified\n");
		ret = -1;
	}

	return ret;
}
