// The protocol between GMM clients and GMM server

#ifndef _GMM_PROTOCOL_H_
#define _GMM_PROTOCOL_H_

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

	char qname_request[16];	// name of msg queue for sending requests to
	char qname_notify[16];	// name of msg queue for sending notifications to
};

// The global management info shared by all GMM clients
struct gmm_global {
	long mem_total;				// total size of device memory
	long mem_used;				// size of used device memory

	int nclients;				// number of attached client processes
	struct gmm_client clients[NCLIENTS];
	int ilru;					// index of the lru client
	int imru;					// index of the mru client
};

#define GMM_SEM_NAME	"/_gmm_semaphore4_"
#define GMM_SHM_NAME	"/_gmm_sharedmem4_"

//#define PAGE_BYTES		4096
//#define PAGE_ALIGNUP(s)	((((s) + PAGE_BYTES - 1) / PAGE_BYTES) * PAGE_BYTES)

#endif
