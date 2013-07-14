#ifndef _GMM_SEG_H_
#define _GMM_SEG_H_

#define GPU_MEM_SIZE 1347483648 //available :2147483648 //Geforce gtx 680

#define GMM_SHARED	9006
#define GMM_SHARED_SIZE	sizeof(gmm_shared_s)

#define GMM_SEM_NAME	"_gmm_semaphore3_"
#define GMM_DEBUG

#include <builtin_types.h>
#include <cuda.h>

#endif
