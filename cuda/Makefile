NVCC=nvcc
LD=$(NVCC)
NVCCFLAGS+=-O3 -arch=sm_60 -std=c++11 -Xcompiler "-Wall -Wextra -Werror"
LDFLAGS+=-lm

OBJS=cg_main.o cg.o matrix.o matrix_coo.o mmio.o vector_gpu.o matrix_gpu.o

all: cgsolver 

cgsolver: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cc
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f cgsolver *.o *~
