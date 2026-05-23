SRC := wfc_openmp.cpp
TARGET := wfc_openmp

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic -fopenmp
LDFLAGS ?= -latomic

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) stb_image.h stb_image_write.h
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
