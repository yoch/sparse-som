TARGET=sparse-som
CXX=g++
CXXFLAGS=--std=c++11 -Wall -Wextra
SRC=../data.cpp ../sparse_vec.cpp ../som.cpp main.cpp


all: som

som:
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -s $(SRC) -o ../$(TARGET)

# (enabling OpenMP is almost useless for now)
omp:
	$(CXX) $(CXXFLAGS) -O2 -s -fopenmp $(SRC) -o ../$(TARGET)-omp

# works well with -march=native
pgo:
	mkdir -p prof
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -march=native -mpc64 -fprofile-generate=./prof $(SRC) -o ../$(TARGET)-prof
	../$(TARGET)-prof -y 15 -x 20 -i ../datasets/libsvm/news20.scale
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -march=native -mpc64 -s -fprofile-use=./prof $(SRC) -o ../$(TARGET)-pgo

clean:
	rm -f *.o
