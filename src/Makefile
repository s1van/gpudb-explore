include gmm.mk

# Specify CUDA install path here
CUDAPATH = /usr/local/cuda

# Name of the GMM shared library
LIBGMM = libgmm.so

# Files needed to create $(LIBGMM)
SRCS = client.c common.c core.c interfaces.c msq.c replacement.c debug.c

OBJS = $(SRCS:.c=.o)

# Use print buffer?
ifeq ($(USE_PRINT_BUFFER), 0)
FLAG_PRINT_BUFFER :=
else
FLAG_PRINT_BUFFER := -DGMM_PRINT_BUFFER
endif

# The compiler/linker settings
CC := gcc
NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -Wall -pthread -fPIC -fvisibility=hidden \
	-I$(CUDAPATH)/include -DCUDAPATH=\"$(CUDAPATH)\" \
	-DGMM_PRINT_LEVEL=$(PRINT_LEVEL) $(FLAG_PRINT_BUFFER) $(GMM_CONFIGS)
LDFLAGS := -shared -pthread -ldl -fPIC

.DEFAULT_GOAL := all
.PHONY : depend all clean install uninstall

all : depend gmmctl $(LIBGMM)

# Generate dependencies for $(OBJS)
depend : .depend

.depend : $(SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > .depend

-include .depend

# No rules for source files
%.c: ;

gmmctl : server.o
	$(NVCC) -L$(CUDAPATH)/lib64 -arch=sm_20 $^ -o $@

server.o : server.c protocol.h spinlock.h list.h atomic.h
	$(NVCC) -c --compiler-options -Wall \
		-I$(CUDAPATH)/include -arch=sm_20 $< -o $@

$(LIBGMM): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	-rm -f gmmctl $(LIBGMM) *.o .depend

# TODO
install: ;

# TODO
uninstall: ;
