include gmm.mk

FLAG_DEBUG = -DGMM_DEBUG
ifeq ($(ENABLE_DEBUG), 0)
FLAG_DEBUG =
endif

FLAG_PROFILE = -DGMM_PROFILE
ifeq ($(ENABLE_PROFILE), 0)
FLAG_PROFILE =
endif

CC = gcc
NVCC = nvcc
# CUDA compile flags
#CUDA_CFLAGS = -I
# CUDA link flags
#CUDA_LDFLAGS = -L
CFLAGS := -O -Wall -pthread -fPIC -fvisibility=hidden \
	$(FLAG_DEBUG) $(FLAG_PROFILE) $(GMM_CONFIGS)
LDFLAGS = -shared -pthread -ldl -fPIC

# Name of the GMM shared library
LIBGMM = libgmm.so

# Object files needed to create $(LIBGMM)
OBJS = common.o core.o interfaces.o msq.o replacement.o

.DEFAULT_GOAL := all
.PHONY : all run purerun clean

all : gmmctl $(LIBGMM) test

gmmctl : server.o
	$(NVCC) -arch=sm_20 $^ -o $@

server.o : server.c protocol.h spinlock.h list.h atomic.h
	$(NVCC) -c -arch=sm_20 $< -o $@

$(LIBGMM): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

# Dependencies with header files
common.o : common.h
core.o: common.h client.h core.h hint.h replacement.h
interfaces.o : common.h core.h protocol.h interfaces.h
msq.o : common.h client.h
replacement.o : replacement.h protocol.h

test:
	$(NVCC) -arch=sm_20 test.cu -o $@

run:
	./gmmctl --start
	LD_PRELOAD=`pwd`/$(LIBGMM) ./test
	./gmmctl --stop

purerun:
	./test

clean:
	-rm -f gmmctl $(LIBGMM) test *.o

# TODO
install: ;

# TODO
uninstall: ;
