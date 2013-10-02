#ifndef _GMM_TEST_H_
#define _GMM_TEST_H_

#include <stdio.h>

struct test_case
{
	int (*func)(void);
	char *comment;
};

#define GMM_TPRINT(fmt, arg...) fprintf(stderr, "[gmm:test] " fmt, ##arg)

#endif
