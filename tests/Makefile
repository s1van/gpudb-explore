# EDIT ME: Test cases
TESTS = test_init_fini test_malloc

# Paths
CUDAPATH = /usr/local/cuda
GMMPATH = ../src
TMPPATH = ./tmp

# Compiler/linker settings
NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -c -O --compiler-options -Wall -arch=sm_20 \
	-I$(CUDAPATH)/include -I$(GMMPATH)
LDFLAGS := -L$(CUDAPATH)/lib64

.DEFAULT_GOAL := all
.SECONDEXPANSION:
.PHONY : all test $(TESTS)

TESTBINS := $(addprefix $(TMPPATH)/,$(TESTS))

all : $(TMPPATH) $(TESTBINS)

$(TMPPATH) :
	@mkdir -p $(TMPPATH)

$(TESTBINS) : $$@.o
	@./tcgen.py $<
	@$(NVCC) $(CFLAGS) main.c -o $(TMPPATH)/main.o
	$(NVCC) $(LDFLAGS) $(TMPPATH)/main.o $< -o $@
	-@rm $(TMPPATH)/main.o

$(TMPPATH)/%.o : %.c
	$(NVCC) $(CFLAGS) $< -o $@

# No rules for source files
%.c : ;

$(TESTS) : $(TMPPATH)/$$@
	$(GMMPATH)/gmmctl --start
	LD_PRELOAD=$(GMMPATH)/libgmm.so $(TMPPATH)/$@
	$(GMMPATH)/gmmctl --stop

test : $(TESTS)

clean:
	-@rm $(TESTBINS) $(TMPPATH)/*.o testcases.h
	-@rm -r $(TMPPATH)
