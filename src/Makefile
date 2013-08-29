CC = gcc
NVCC = nvcc

all : gmm_server gmm_client ic_gmm.so test

gmm_server : gmm_server.o gmm_core.o
	$(CC) -pthread -ldl $(.SOURCES) -o $(.TARGET)

gmm_server.o : gmm_server.c
	$(CC) -pthread -ldl -fPIC -c $(.SOURCE) -o $(.TARGET)

gmm_client: gmm_client.c gmm_core.c
	$(CC) -pthread -ldl -fPIC gmm_client.c gmm_core.c -o $(.TARGET)

ic_gmm.so: intercept_gmm.c gmm_core.c
	$(CC) -shared -ldl -fPIC intercept_gmm.c gmm_core.c -o $(.TARGET)

test: 
	$(NVCC) test.cu -arch=sm_20 -o $(.TARGET)

.PHONY: all run purerun clean

run:
	LD_PRELOAD=`pwd`/ic_gmm.so ./test

pure_run:
	./test

clean:
	rm -f gmm_server gmm_client ic_gmm.so test
