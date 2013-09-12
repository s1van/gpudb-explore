// The protocol between GMM clients and GMM server

#ifndef _GMM_PROTOCOL_H_
#define _GMM_PROTOCOL_H_

#include <unistd.h>
#include "list.h"
#include "spinlock.h"
#include "atomic.h"

// The maximum number of concurrent processes managed by GMM
#define NCLIENTS	32

// A GMM client registered in the global shared memory
struct gmm_client {
	int index;				// index of this client; -1 means unoccupied
	struct {
		int prev;
		int next;
	} list_client;			// linked into the LRU client list

	//long size_attached;		// TODO: maybe hide this?
	long size_detachable;
	//long lru_size;		// TODO
	//long lru_cost;		// TODO

	// Each client has two message queues: one that receives
	// requests from other clients, named "gr_%pid", the other
	// that receives notifications, named "gn_%pid"
	pid_t pid;				// pid of the client process
};

// The global management info shared by all GMM clients
struct gmm_global {
	struct spinlock lock;		// The lock; it works only when the cache
								// coherence protocol works for virutal caches
								// as well.
	long mem_total;				// Total size of device memory
	atomic_l_t mem_used;		// Size of used (attached) device memory
								// NOTE: in numbers, device memory may be
								// over-used, i.e., mem_used > mem_total

	int maxclients;				// Max number of clients supported
	int nclients;				// Number of attached client processes
	int ilru;					// Index of the LRU client
	int imru;					// Index of the MRU client
	struct gmm_client clients[NCLIENTS];
};

#define GMM_SEM_NAME	"/gmm_semaphore"
#define GMM_SHM_NAME	"/gmm_sharedmem"

// Add the inew'th client to the MRU end of p's client list
static inline void ILIST_ADD(struct gmm_global *p, int inew)
{
	if (p->imru == -1) {
		p->ilru = p->imru = inew;
		p->clients[inew].list_client.prev = -1;
		p->clients[inew].list_client.next = -1;
	}
	else {
		p->clients[inew].list_client.prev = -1;
		p->clients[inew].list_client.next = p->imru;
		p->clients[p->imru].list_client.prev = inew;
		p->imru = inew;
	}
}

// Delete a client from p's client list
static inline void ILIST_DEL(struct gmm_global *p, int idel)
{
	int iprev = p->clients[idel].list_client.prev;
	int inext = p->clients[idel].list_client.next;

	if (iprev != -1)
		p->clients[iprev].list_client.next = inext;
	else
		p->imru = inext;

	if (inext != -1)
		p->clients[inext].list_client.prev = iprev;
	else
		p->ilru = iprev;
}

#endif
