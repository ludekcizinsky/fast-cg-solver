OPTIM = -O1

CXX = mpicxx
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -Werror -pedantic -std=c++11 -I${OPENBLAS_ROOT}/include -g $(OPTIM)
LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lopenblas

OBJS=cg_mpi_main.o cg_mpi.o matrix_coo.o mmio.o

all: cg_mpi

cg_mpi: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f cg_mpi *.o *~