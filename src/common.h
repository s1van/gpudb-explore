#ifndef _GMM_COMMON_H_
#define _GMM_COMMON_H_

#include <stdio.h>
#include <execinfo.h>

#define GMM_EXPORT __attribute__((__visibility__("default")))

#define FATAL	0
#define ERROR	1
#define WARN	2
#define INFO	3
#define DEBUG	4
#define PRINT_LEVELS	5

#ifndef GMM_PRINT_LEVEL
#define GMM_PRINT_LEVEL		FATAL
#endif

extern char *GMM_PRINT_MSG[];

#ifdef GMM_PRINT_BUFFER
#include <sys/time.h>
#include "spinlock.h"

extern char *gprint_buffer;
extern int gprint_lines;
extern int gprint_head;
extern struct spinlock gprint_lock;

#define gprint(lvl, fmt, arg...) \
	do { \
		if (lvl <= GMM_PRINT_LEVEL) { \
			int len; \
			struct timeval t; \
			gettimeofday(&t, NULL); \
			acquire(&gprint_lock); \
			len = sprintf(gprint_buffer + gprint_head, \
					"[%d %s] %lf " fmt, getpid(), GMM_PRINT_MSG[lvl], \
					((double)(t.tv_sec) + t.tv_usec / 1000000.0), ##arg); \
			gprint_lines++; \
			gprint_head += len + 1; \
			release(&gprint_lock); \
		} \
	} while (0)
#else
#define gprint(lvl, fmt, arg...) \
		do { \
			if (lvl <= GMM_PRINT_LEVEL) { \
				printf("[%d %s] " fmt, getpid(), GMM_PRINT_MSG[lvl], ##arg); \
			} \
		} while (0)
#endif

void gprint_init();
void gprint_fini();

#ifndef gettid
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
static inline pid_t gettid()
{
	return (pid_t)syscall(186);
}
#endif

void panic(char *msg);

#endif
