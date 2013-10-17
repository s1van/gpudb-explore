# EDIT ME: Test cases
TESTS = test_spinlock test_init_fini test_malloc test_memcpy test_launch

# Paths
CUDAPATH = /usr/local/cuda
GMMPATH := `pwd`/../src
TMPPATH = ./tmp

# Compiler/linker settings
NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -c --compiler-options -Wall -arch=sm_20 \
	-I$(CUDAPATH)/include -I$(GMMPATH)
LDFLAGS := -L$(CUDAPATH)/lib64 -L$(GMMPATH) -Xlinker -rpath=$(GMMPATH) -lgmm

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
	@$(GMMPATH)/gmmctl --start -v
	@echo ""

cleanup:
	@$(GMMPATH)/gmmctl --stop -v

clean:
	-@rm $(TESTBINS) $(TMPPATH)/*.o testcases.h
	-@rm -r $(TMPPATH)
