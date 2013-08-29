#ifndef _GMM_TYPE_H_
#define _GMM_TYPE_H_

#include <pthread.h>


// The maximum number of concurrent processes managed by GMM
#define NCLIENTS	32

// The global management info shared by all GMM clients
struct gmm_global {
	int nclients;				// number of attached client processes
	int next_id;				// id for next instance
	long mem_free;
	long claimed[NCLIENTS];	//memory usage: f(id) -> index
	long in_gpu[NCLIENTS];
	long in_use[NCLIENTS];	//in_gpu_mem >= in_use_mem
	long swapped[NCLIENTS];	//claimed == swapped + in_gpu_mem
	int wait[NCLIENTS];		//waiting list: f(id) -> index: 0(false), id(true)
};

// The local management info within each GMM client
struct gmm_local {
	pthread_mutex_t mutex;		// mutex for local synchronizations
	cfuhash_table_t *all;
	cfuhash_table_t *in_gpu_mem;
	cfuhash_table_t *in_use;
	cfuhash_table_t *swapped;
	cfuhash_table_t *unmalloced;
	int count;			//number of objects
};

typedef enum {
	in_gpu_mem = 0,
	swapped,
	in_use,
	unmalloced,
	outlaw
} objState;

typedef struct{
	void *key;	//generated randomly
	void *devPtr;	//pointer to object in device
	void *memPtr;	//pointer to object in main memory
	size_t size;	//size of the object in bytes
	objState state;
	int in_use_count; //TODO
} *gmm_obj, gmm_obj_s;

#define GMM_OBJ_SIZE sizeof(gmm_obj_s)
#define GMM_KEY_SIZE sizeof(void *)


#define GMM_KERNEL_ARG_NUM_MAX	16
typedef struct{
	gmm_obj in_use[GMM_KERNEL_ARG_NUM_MAX];
	int in_use_num;	
} *gmm_args, gmm_args_s;

#endif
