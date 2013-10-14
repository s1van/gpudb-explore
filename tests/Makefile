# EDIT ME: Test cases
TESTS = test_init_fini test_malloc

# Paths
CUDAPATH = /usr/local/cuda
GMMPATH = ../src
TMPPATH = ./tmp

# Compiler/linker settings
NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -c --compiler-options -Wall -arch=sm_20 \
	-I$(CUDAPATH)/include -I$(GMMPATH)
LDFLAGS := -L$(CUDAPATH)/lib64

.DEFAULT_GOAL := all
.SECONDEXPANSION:
.PHONY : all test setup cleanup $(TESTS)

TESTBINS := $(addprefix $(TMPPATH)/,$(TESTS))

all : $(TMPPATH) $(TESTBINS)

$(TMPPATH) :
	@mkdir -p $(TMPPATH)

$(TESTBINS) : $$@.o
	@./tcgen.py $<
	@$(NVCC) $(CFLAGS) main.cu -o $(TMPPATH)/main.o
	$(NVCC) $(LDFLAGS) $(TMPPATH)/main.o $< -o $@
	-@rm $(TMPPATH)/main.o

$(TMPPATH)/%.o : %.cu
	$(NVCC) $(CFLAGS) $< -o $@

# No rules for source files
%.c : ;

$(TESTS) : $(TMPPATH)/$$@
	@echo "================================================================"
	@LD_PRELOAD=$(GMMPATH)/libgmm.so $(TMPPATH)/$@
	@echo ""

test : setup $(TESTS) cleanup

setup:
	@$(GMMPATH)/gmmctl --start
	@echo ""

cleanup:
	@$(GMMPATH)/gmmctl --stop

clean:
	-@rm $(TESTBINS) $(TMPPATH)/*.o testcases.h
	-@rm -r $(TMPPATH)
