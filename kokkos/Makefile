SRC := wfc_kokkos.cpp
TARGET := wfc_kokkos
BACKEND ?= openmp
BUILD_DIR ?= build/$(BACKEND)

VALID_BACKENDS := cpu cuda hip openmp

ifeq ($(filter $(BACKEND),$(VALID_BACKENDS)),)
  $(error BACKEND must be one of: $(VALID_BACKENDS))
endif

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -pthread -ldl -latomic

KOKKOS_PREFIX ?=
KOKKOS_CFLAGS ?= $(shell pkg-config --cflags kokkos 2>/dev/null)
KOKKOS_LIBS ?= $(shell pkg-config --libs kokkos 2>/dev/null)

ifeq ($(strip $(KOKKOS_CFLAGS)),)
  ifneq ($(strip $(KOKKOS_PREFIX)),)
    KOKKOS_CFLAGS := -I$(KOKKOS_PREFIX)/include
  endif
endif

ifeq ($(strip $(KOKKOS_LIBS)),)
	KOKKOS_LIBS := -lkokkoscore -lkokkoscontainers
	ifneq ($(strip $(KOKKOS_PREFIX)),)
		KOKKOS_LIBS := -L$(KOKKOS_PREFIX)/lib $(KOKKOS_LIBS)
	endif
endif

ifeq ($(BACKEND),cuda)
	ifeq ($(origin CXX), default)
		override CXX := nvcc_wrapper
	endif
  BACKEND_CXXFLAGS :=
	  BACKEND_LDFLAGS :=
else ifeq ($(BACKEND),hip)
	ifeq ($(origin CXX), default)
		override CXX := hipcc
	endif
  BACKEND_CXXFLAGS :=
	  BACKEND_LDFLAGS :=
else ifeq ($(BACKEND),openmp)
	ifeq ($(origin CXX), default)
		override CXX := c++
	endif
  BACKEND_CXXFLAGS := -fopenmp
	  BACKEND_LDFLAGS := -fopenmp
else
	ifeq ($(origin CXX), default)
		override CXX := c++
	endif
	  BACKEND_CXXFLAGS := -fopenmp
	  BACKEND_LDFLAGS := -fopenmp
endif

OBJ := $(BUILD_DIR)/$(TARGET).o
BIN := $(BUILD_DIR)/$(TARGET)
DEPS := $(OBJ:.o=.d)

.PHONY: all run clean distclean cpu cuda hip openmp help clean-output

all: $(BIN)

$(BUILD_DIR):
	@mkdir -p "$@"

$(OBJ): $(SRC) stb_image.h stb_image_write.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(KOKKOS_CFLAGS) $(CXXFLAGS) $(BACKEND_CXXFLAGS) -MMD -MP -c $(SRC) -o $@

$(BIN): $(OBJ)
	$(CXX) $(LDFLAGS) $(BACKEND_LDFLAGS) $^ $(KOKKOS_LIBS) $(LDLIBS) -o $@

run: $(BIN)
	./$(BIN)

cpu:
	$(MAKE) BACKEND=cpu all

cuda:
	$(MAKE) BACKEND=cuda all

hip:
	$(MAKE) BACKEND=hip all

openmp:
	$(MAKE) BACKEND=openmp all

clean:
	rm -rf build

distclean: clean

clean-output:
	rm -f output.png output.ppm

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make                Build with BACKEND=openmp (default)' \
	  '  make BACKEND=cpu    Build the CPU/serial variant' \
	  '  make BACKEND=openmp Build the OpenMP variant' \
	  '  make BACKEND=cuda   Build the CUDA variant' \
	  '  make BACKEND=hip    Build the HIP variant' \
	  '  make run            Build then run the selected backend' \
	  '  make clean          Remove build artifacts' \
	  'Variables:' \
	  '  BACKEND=cpu|cuda|hip|openmp' \
	  '  KOKKOS_PREFIX=/path/to/kokkos/install' \
	  '  CXX=...             Override the compiler or wrapper' \
	  '  OMP_NUM_THREADS=N    Set OpenMP threads at runtime'

-include $(DEPS)