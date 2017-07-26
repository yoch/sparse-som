TARGET=sparse-bsom
CXX=g++
CXXFLAGS=--std=c++11 -Wall -Wextra
SRC=../data.cpp ../sparse_vec.cpp ../bsom.cpp main.cpp

all: bsom omp


bsom:
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -s $(SRC) -o ../$(TARGET)	

omp:
	$(CXX) $(CXXFLAGS) -O2 -s -fopenmp $(SRC) -o ../$(TARGET)-omp

#-mpc32 -march=native -mtune=native -ffast-math

prof:
	$(CXX) $(CXXFLAGS) -Og -pg -fopenmp $(SRC) -o ../$(TARGET)-omp-prof

# works well with -march=native
pgo:
	mkdir -p prof
	$(CXX) $(CXXFLAGS) -O2  -fopenmp -march=native -mpc64 -fprofile-generate=./prof $(SRC) -o ../$(TARGET)-prof
	../$(TARGET)-prof -y 15 -x 20 -i ../datasets/libsvm/news20.scale
	$(CXX) $(CXXFLAGS) -O2  -fopenmp -march=native -mpc64 -s -fprofile-use=./prof -fprofile-correction $(SRC) -o ../$(TARGET)-pgo

clean:
	rm -f *.o
