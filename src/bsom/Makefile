TARGET=sparse-bsom
CXX=g++
CXXFLAGS=--std=c++11 -Wall -Wextra
SRC=../data.cpp ../bsom.cpp main.cpp
BUILDDIR=../../build
DATADIR=../../datasets


all: bsom


bsom:
#	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -O2 -s $(SRC) -o $(BUILDDIR)/$(TARGET)	
	$(CXX) $(CXXFLAGS) -O2 -s -fopenmp $(SRC) -o $(BUILDDIR)/$(TARGET)

#-mpc32 -march=native -mtune=native -ffast-math

prof:
	$(CXX) $(CXXFLAGS) -Og -pg -fopenmp $(SRC) -o $(BUILDDIR)/$(TARGET)-omp-prof

# works well with -march=native
pgo:
	mkdir -p $(BUILDDIR)/prof
	$(CXX) $(CXXFLAGS) -O2 -fopenmp -march=native -mpc64 -fprofile-generate=$(BUILDDIR)/prof $(SRC) -o $(BUILDDIR)/$(TARGET)-prof
	$(BUILDDIR)/$(TARGET)-prof -q -y 15 -x 20 -i $(DATADIR)/news20.scale
	$(CXX) $(CXXFLAGS) -O2 -fopenmp -march=native -mpc64 -s -fprofile-use=$(BUILDDIR)/prof -fprofile-correction $(SRC) -o $(BUILDDIR)/$(TARGET)-pgo

clean:
	rm -f *.o
