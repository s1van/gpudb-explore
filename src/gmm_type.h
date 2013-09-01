#ifndef _GMM_TYPE_H_
#define _GMM_TYPE_H_

#include <pthread.h>
#include <mqueue.h>
#include "gmm_list.h"


// The local management info within each GMM client
struct gmm_local {
	pthread_mutex_t mutex;		// mutex for local synchronizations
	struct list_head all;
	struct list_head attached;
	int count;
};

// State of a device memory object
enum memobj_state {
	STATE_ATTACHED = 0,		// ATTACHED: object w/ dev mem allocated
	STATE_DETACHED,			// DETACHED: object w/o dev mem allocated
	STATE_FREEING,
	STATE_EVICTING,
	STATE_EVICTED,
	STATE_OUTLAW
};

// Device memory object
struct memobj {
	long size;					// size of the object in bytes
	enum memobj_state state;	// state of the object
	void *addr_dev;				// pointer to object in device
	void *addr_sys;				// pointer to object in main memory
	int pinned;					// pin counter
};

/*
#define GMM_KERNEL_ARG_NUM_MAX	16
typedef struct{
	gmm_obj in_use[GMM_KERNEL_ARG_NUM_MAX];
	int in_use_num;	
} *gmm_args, gmm_args_s;
*/

#endif
