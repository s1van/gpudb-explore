#ifndef _GMM_COMMON_H_
#define _GMM_COMMON_H_

#include <stdio.h>
#include <execinfo.h>

#define GMM_EXPORT __attribute__((__visibility__("default")))

#ifdef GMM_DEBUG
#define GMM_DPRINT(fmt, arg...) fprintf(stderr, "[gmm:debug] " fmt, ##arg)
#else
#define GMM_DPRINT(fmt, arg...)
#endif

#ifdef GMM_PROFILE
#define GMM_PRINT(fmt, arg...) fprintf(stderr, "[gmm:profile] " fmt, ##arg)
#else
#define GMM_PRINT(fmt, arg...)
#endif

static inline void show_stackframe() {
  void *trace[32];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  trace_size = backtrace(trace, 32);
  messages = backtrace_symbols(trace, trace_size);
  fprintf(stderr, "Printing stack frames:\n");
  for (i=0; i < trace_size; ++i)
        fprintf(stderr, "\t%s\n", messages[i]);
}

static inline void panic(char *msg)
{
	fprintf(stderr, "[gmm:panic] %s\n", msg);
	show_stackframe();
	exit(-1);
}

#endif
