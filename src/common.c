#include <stdio.h>
#include <stdlib.h>
#include "common.h"

char *GMM_PRINT_MSG[PRINT_LEVELS] = {
		"fatal",
		"error",
		" warn",
		" info",
		"debug"
};

static void show_stackframe() {
  void *trace[32];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  trace_size = backtrace(trace, 32);
  messages = backtrace_symbols(trace, trace_size);
  fprintf(stderr, "Printing stack frames:\n");
  for (i=0; i < trace_size; ++i)
        fprintf(stderr, "\t%s\n", messages[i]);
}

void panic(char *msg)
{
	fprintf(stderr, "[gmm:panic] %s\n", msg);
	show_stackframe();
	exit(-1);
}

#ifdef GMM_PRINT_BUFFER
char *gprint_buffer = NULL;
#define GBUFFER_SIZE		(128L * 1000L * 1000L)
int gprint_lines = 0;
int gprint_head = 0;
struct spinlock gprint_lock;
#endif

void gprint_init()
{
#ifdef GMM_PRINT_BUFFER
	int i;

	initlock(&gprint_lock);

	gprint_buffer = (char *)malloc(GBUFFER_SIZE);
	if (!gprint_buffer) {
		fprintf(stderr, "failed to initialize gprint buffer\n");
		exit(-1);
	}

	for (i = 0; i < GBUFFER_SIZE; i += 4096)
		gprint_buffer[i] = 'x';
#endif
}

void gprint_fini()
{
#ifdef GMM_PRINT_BUFFER
	int i, head = 0, len;
	if (gprint_buffer) {
		for (i = 0; i < gprint_lines; i++) {
			len = printf("%s", gprint_buffer + head);
			head += len + 1;
		}
		free(gprint_buffer);
		gprint_buffer = NULL;
	}
#endif
}
