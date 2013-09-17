#include <fcntl.h>
#include <semaphore.h>
#include <mqueue.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#include "protocol.h"


sem_t sem_attach = SEM_FAILED;
struct gmm_global *pglobal = NULL;
int cid = -1;

mqd_t mqid_request = (mqd_t) -1;
pthread_t tid_request;
mqd_t mqid_notify = (mqd_t) -1;
//pthread_t tid_notify;		// we omit the notification thread for now

int attached = 0;


// The thread that listens to the request message queue
void *thread_request_listener(void *arg)
{
	return NULL;
}

// Free the client slot taken by the calling process
static void free_client(int id)
{
	char qname[16];

	if (mqid_request != (mqd_t) -1) {
		sprintf(qname, "/gr_%d", pglobal->clients[id].pid);
		mq_unlink(qname);
		mq_close(mqid_request);
		mqid_request = (mqd_t) -1;
		// TODO: destroy the request listener thread
	}
	if (mqid_notify != (mqd_t)-1) {
		sprintf(qname, "/gn_%d", pglobal->clients[id].pid);
		mq_unlink(qname);
		mq_close(mqid_notify);
		mqid_notify = (mqd_t)-1;
	}

	// Remove this client from global client list
	acquire(&pglobal->lock);
	ILIST_DEL(pglobal, id);
	pglobal->nclients--;
	//memset(pglobal->clients + id, 0, sizeof(pglobal->clients[0]));
	pglobal->clients[id].index = -1;
	release(&pglobal->lock);
}

// Allocate a new client slot for the calling process
static int alloc_client()
{
	char qname[16];
	int id = -1;

	// Get a unique client id
	acquire(&pglobal->lock);
	if (pglobal->nclients < pglobal->maxclients) {
		for (id = 0; id < pglobal->maxclients; id++) {
			if (pglobal->clients[id].index == -1) {
				break;
			}
		}
	}
	if (id >= 0 && id < pglobal->maxclients) {
		memset(pglobal->clients + id, 0, sizeof(pglobal->clients[0]));
		pglobal->clients[id].index = id;
		pglobal->clients[id].pid = getpid();
		ILIST_ADD(pglobal, id);
		pglobal->nclients++;
	}
	release(&pglobal->lock);

	// Allocate request and notification message queues
	if (id >= 0 && id < pglobal->maxclients) {
		sprintf(qname, "/gr_%d", getpid());
		mqid_request = mq_open(qname, O_RDONLY | O_CREAT | O_EXCL, 0622, NULL);
		if (mqid_request == (mqd_t) -1) {
			GMM_DPRINT("failed to create the request queue: %s\n", \
					strerror(errno));
			free_client(id);
			return -1;
		}
		if (pthread_create(&tid_request, NULL, thread_request_listener,
				NULL) != 0) {
			mq_unlink(mqid_request);
			mq_close(mqid_request);
			mqid_request = (mqd_t) -1;
			free_client(id);
			return -1;
		}

		sprintf(qname, "/gn_%d", getpid());
		mqid_notify = mq_open(qname, O_RDONLY | O_CREAT | O_EXCL, 0622, NULL);
		if (mqid_notify == (mqd_t) -1) {
			GMM_DPRINT("failed to create the notification queue: %s\n", \
					strerror(errno));
			free_client(id);
			return -1;
		}
	}

	return id;
}

// Attach this process to the global GMM arena.
int client_attach()
{
	int shmfd;

	// Get the semaphore
	sem_attach = sem_open(GMM_SEM_NAME, 0);
	if (sem_attach == SEM_FAILED) {
		GMM_DPRINT("unable to open semaphore: %s\n", strerror(errno));
		return -1;
	}

	// Get the shared memory
	shmfd = shm_open(GMM_SHM_NAME, O_RDWR, 0);
	if (shmfd == -1) {
		GMM_DPRINT("unable to open shared memory: %s\n", strerror(errno));
		goto fail_shm;
	}

	pglobal = (struct gmm_global *)mmap(NULL, sizeof(*pglobal),
			PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
	if (pglobal == MAP_FAILED) {
		GMM_DPRINT("failed to mmap the shared memory: %s\n", strerror(errno));
		goto fail_mmap;
	}
	// TODO: resize the mapping according to pglobal->maxclients

	// Allocate a new client slot
	cid = alloc_client();
	if (cid == -1) {
		GMM_DPRINT("failed to allocate client\n");
		cid = -1;
		goto fail_client;
	}

	attached = 1;
	close(shmfd);
	return 0;

fail_client:
	munmap(pglobal, sizeof(*pglobal));
fail_mmap:
	pglobal = NULL;
	close(shmfd);
fail_shm:
	sem_close(GMM_SEM_NAME);
	sem_attach = SEM_FAILED;

	return -1;
}

// TODO: have to make sure operations are executed only when attach was
// successful
void client_detach() {
	attached = 0;
	free_client();
	cid = -1;
	if (pglobal != NULL) {
		munmap(pglobal, sizeof(*pglobal));
		pglobal = NULL;
	}
	if (sem_attach != SEM_FAILED) {
		sem_close(GMM_SEM_NAME);
		sem_attach = SEM_FAILED;
	}
}

long memsize()
{
	return pglobal->mem_total;
}

long free_memsize()
{
	long freesize = pglobal->mem_total - pglobal->mem_used;
	return freesize < 0 ? 0 : freesize;
}

long free_memsize2()
{
	return pglobal->mem_total - pglobal->mem_used;
}

void update_attached(long delta)
{
	atomic_addl(&pglobal->mem_used, delta);
}

void update_detachable(long delta)
{
	atomic_addl(&pglobal->clients[cid].size_detachable, delta);
}

void begin_load()
{
	int ret;
	do {
		ret = sem_wait(&sem_attach);
	} while (ret == -1 && errno == EINTR);
}

void end_load()
{
	sem_post(&sem_attach);
}

// Get the id of the least recently used client with detachable device memory.
// The client is pinned if it is a remote client.
int client_lru_detachable()
{
	int iclient;

	acquire(&pglobal->lock);
	for (iclient = pglobal->ilru; iclient != -1;
			iclient = pglobal->clients[iclient].iprev) {
		if (pglobal->clients[iclient].size_detachable > 0)
			break;
	}
	if (iclient != -1 && iclient != cid)
		pglobal->clients[iclient].pin++;
	release(&pglobal->lock);

	return iclient;
}

void client_unpin(int client)
{
	acquire(&pglobal->lock);
	pglobal->clients[client].pin--;
	release(&pglobal->lock);
}

// Is the client a local client?
int is_client_local(int client)
{
	return client == cid;
}

int remote_victim_evict(int client, long size_needed)
{
	return 0;
}
