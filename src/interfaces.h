#ifndef _GMM_INTERFACES_H_
#define _GMM_INTERFACES_H_

#define __USE_GNU		// _GNU_SOURCE
#include <dlfcn.h>		/* header required for dlopen() and dlsym() */
#include <stdio.h>
#include <stdlib.h>

//#define GET_TIMEVAL(_t) (_t.tv_sec + _t.tv_usec / 1000000.0)
//#define GMM_IGNORE_SIZE 5038400

#ifdef CUDAPATH
#define CUDA_CU_PATH    (CUDAPATH "/lib64/libcuinj64.so")
#define CUDA_CURT_PATH  (CUDAPATH "/lib64/libcudart.so")
#else
#define CUDA_CU_PATH    "/usr/local/cuda/lib64/libcuinj64.so"
#define CUDA_CURT_PATH  "/usr/local/cuda/lib64/libcudart.so"
#endif

/* Check whether dlsym returned successfully */
#define  TREAT_ERROR()                          \
  do {                                          \
    char * __error;                             \
    if ((__error = dlerror()) != NULL) {        \
      fputs(__error, stderr);                   \
      abort();                                  \
    }                                           \
  }while(0)

/* Intercept function func and store its previous value into var */
#define INTERCEPT_CUDA(func, var)       \
do {                                    \
        if(var) break;                  \
        var = (typeof(var))dlsym(RTLD_NEXT, func);  \
        TREAT_ERROR();                  \
} while(0)

#define INTERCEPT_CUDA2(func, var)	\
do {                                    \
        if(var) break;                  \
        void *__handle = dlopen(CUDA_CURT_PATH, RTLD_LOCAL | RTLD_LAZY);    \
        var = (typeof(var))dlsym(__handle, func);  \
        TREAT_ERROR();                  \
} while(0)

#endif
