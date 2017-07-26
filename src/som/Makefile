TARGET=sparse-som
CXX=g++
CXXFLAGS=--std=c++11 -Wall -Wextra
SRC=../data.cpp ../sparse_vec.cpp ../som.cpp main.cpp
BUILDDIR=../../build
DATADIR=../../datasets


all: som

som:
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -s $(SRC) -o $(BUILDDIR)/$(TARGET)

# (enabling OpenMP is almost useless for now)
omp:
	$(CXX) $(CXXFLAGS) -O2 -s -fopenmp $(SRC) -o $(BUILDDIR)/$(TARGET)-omp

# works well with -march=native
pgo:
	mkdir -p $(BUILDDIR)/prof
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -march=native -mpc64 -fprofile-generate=$(BUILDDIR)/prof $(SRC) -o $(BUILDDIR)/$(TARGET)-prof
	$(BUILDDIR)/$(TARGET)-prof -q -y 15 -x 20 -i $(DATADIR)/news20.scale
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -march=native -mpc64 -s -fprofile-use=$(BUILDDIR)/prof $(SRC) -o $(BUILDDIR)/$(TARGET)-pgo

clean:
	rm -f *.o
