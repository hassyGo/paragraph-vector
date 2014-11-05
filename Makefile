CXX=g++

EIGEN_LOCATION=$$HOME/local/eigen #Change this line
BUILD_DIR=./

CXXFLAGS =-Wall
CXXFLAGS+=-O3
CXXFLAGS+=-std=c++0x
CXXFLAGS+=-lm
CXXFLAGS+=-fomit-frame-pointer
CXXFLAGS+=-fno-schedule-insns2
CXXFLAGS+=-fexceptions
CXXFLAGS+=-funroll-loops
CXXFLAGS+=-march=native
CXXFLAGS+=-mfpmath=sse
CXXFLAGS+=-fopenmp
CXXFLAGS+=-m64
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
CXXFLAGS+=-DEIGEN_NO_DEBUG
CXXFLAGS+=-DEIGEN_NO_STATIC_ASSERT
CXXFLAGS+=-I$(EIGEN_LOCATION)

CXXLIBS=-fopenmp

SRCS=$(shell ls *.cpp)
OBJS=$(SRCS:.cpp=.o)

PROGRAM=paragraph_vector

all : $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,$(PROGRAM))

$(BUILD_DIR)/%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/$(PROGRAM) : $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
	$(CXX) $(CXXFLAGS) $(CXXLIBS) -o $@ $^
	rm -f ?*~

clean:
	rm -f $(BUILD_DIR)/*.o $(PROGRAM) ?*~
