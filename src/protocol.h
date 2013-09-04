// The protocol between GMM clients and GMM server

#ifndef _GMM_PROTOCOL_H_
#define _GMM_PROTOCOL_H_

#include <unistd.h>
#include "gmm_list.h"

// The maximum number of concurrent processes managed by GMM
#define NCLIENTS	32

// A GMM client registered in the global shared memory
struct gmm_client {
	int index;				// index of this client; -1 means unoccupied
	struct {
		int prev;
		int next;
	} list_client;			// linked into the LRU client list

	long size_attached;
	long size_detachable;
	long lru_size;
	long lru_cost;

	// Each client has two message queues: one that receives
	// requests from other clients, named "gr_%pid", the other
	// that receives notifications, named "gn_%pid"
	pid_t pid;				// pid of the client process
};

// The global management info shared by all GMM clients
struct gmm_global {
	long mem_total;				// total size of device memory
	long mem_used;				// size of used device memory

	int maxclients;				// max number of clients supported
	int nclients;				// number of attached client processes
	int ilru;					// index of the lru client
	int imru;					// index of the mru client
	struct gmm_client clients[NCLIENTS];
};

#define GMM_SEM_NAME	"/_gmm_semaphore4_"
#define GMM_SHM_NAME	"/_gmm_sharedmem4_"

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
