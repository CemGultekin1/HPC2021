CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.c*))

all : $(TARGETS)

%:%.c
	$(CXX) $(CXXFLAGS) $^ -o $@
%:%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)

