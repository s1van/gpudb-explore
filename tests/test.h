#ifndef _GMM_TEST_H_
#define _GMM_TEST_H_

struct test_case
{
	int (*func)(void);
	char *comment;
};

#endif
