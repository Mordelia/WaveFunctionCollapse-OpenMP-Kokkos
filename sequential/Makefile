SRC := wfc.cpp
TARGET := wfc

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS ?=

.PHONY: all run clean clean-output

all: $(TARGET)

$(TARGET): $(SRC) stb_image.h stb_image_write.h
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean: clean-output
	rm -f $(TARGET)

clean-output:
	rm -f output.png output.ppm
