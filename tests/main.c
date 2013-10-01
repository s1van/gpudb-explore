// Driver of test cases:
// For each test case, create a separate source file, e.g., test_demo.c.
// In test_demo.c, the first few comment lines are some description
// texts about the test case, which will be printed when the test case
// is executed. Following the comment lines is the test function, whose
// name must match the name of the test file (in this case, int test_demo()).
// After test_demo.c is completed, register it in the Makefile under the
// TESTS variable. All test cases can be compiled and ran with command:
//			make && make test
// Note: testcases.h is generated automatically by tcgen.py for each test case.
#include <stdio.h>
#include "testcases.h"

int main()
{
	int ncases = sizeof(testcases) / sizeof(testcases[0]);
	int i, ret = 0, tret = 0;

	for (i = 0; i < ncases; i++) {
		fprintf(stderr, "Testing %s\n", testcases[i].comment);
		tret = testcases[i].func();
		if (tret == 0)
			fprintf(stderr, "Test passed\n\n");
		else {
			fprintf(stderr, "Test failed: %d\n\n", tret);
			ret = -1;
		}
	}

	return ret;
}
