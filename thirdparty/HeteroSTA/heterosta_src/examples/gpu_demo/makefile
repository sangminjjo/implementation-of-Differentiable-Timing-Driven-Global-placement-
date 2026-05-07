# ======================================
# Makefile for the HeteroSTA C++ Demos
# ======================================

# --- Compiler and Targets ---
CXX := g++
RM  := rm -f
TARGETS := read_netlist_demo multi_gpu_demo set_netlistdb_demo

# --- Source and Object Files ---
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
EXTRA_OBJECTS := placedb.o
CLEAN_FILES := $(TARGETS) $(OBJECTS) $(EXTRA_OBJECTS)

# --- Flags ---
CXXFLAGS := -std=c++17 -g -Wall -fPIC
LDFLAGS  :=
LDLIBS   := -lheterosta -lgomp -lrt -lcuda -lcudart -pthread -ldl

# --- Paths (USER CONFIGURABLE for HeteroSTA) ---
HETEROSTA_ROOT ?= ../..
HETEROSTA_INC_DIR ?= $(HETEROSTA_ROOT)/include
HETEROSTA_LIB_DIR ?= $(HETEROSTA_ROOT)/lib
COMMON_DIR := ../common
vpath %.cpp $(COMMON_DIR)


NEEDS_BUILD := $(filter $(MAKECMDGOALS),all $(TARGETS))
ifeq ($(strip $(NEEDS_BUILD)),)
    IS_BUILD_COMMAND :=
else
    IS_BUILD_COMMAND := true
endif
ifeq ($(strip $(MAKECMDGOALS)),)
    IS_BUILD_COMMAND := true
endif

ifeq ($(IS_BUILD_COMMAND),true)

ifndef CUDA_HOME
    ifneq ($(strip $(CONDA_PREFIX)),)
        ifneq ($(wildcard $(CONDA_PREFIX)/include/cuda_runtime.h),)
            DETECTED_CUDA_HOME := $(CONDA_PREFIX)
            INFO_MSG_CUDA_TOOLKIT := "CUDA Toolkit:  Found in Conda prefix -> $(CONDA_PREFIX)"
        endif
    endif
    ifndef DETECTED_CUDA_HOME
        NVCC_PATH := $(shell which nvcc)
        ifneq ($(NVCC_PATH),)
            DETECTED_CUDA_HOME := $(shell dirname $(shell dirname $(NVCC_PATH)))
            INFO_MSG_CUDA_TOOLKIT := "CUDA Toolkit:  Found via nvcc path -> $(DETECTED_CUDA_HOME)"
        endif
    endif
    ifndef DETECTED_CUDA_HOME
        DETECTED_CUDA_HOME := /usr/local/cuda
        INFO_MSG_CUDA_TOOLKIT := "CUDA Toolkit:  Fallback to default -> /usr/local/cuda"
    endif
    CUDA_HOME := $(DETECTED_CUDA_HOME)
else
    INFO_MSG_CUDA_TOOLKIT := "CUDA Toolkit:  Using user-provided CUDA_HOME -> $(CUDA_HOME)"
endif

# --- Automatic System Library Detection ---
CUDA_DRIVER_LIB_PATH := $(shell ldconfig -p | grep -o '/.*libcuda\.so' | head -n 1 | xargs -r dirname)
ifeq ($(strip $(CUDA_DRIVER_LIB_PATH)),)
    CUDA_DRIVER_LIB_PATH := $(shell find /usr/lib /usr/lib64 -name "libcuda.so" -print -quit 2>/dev/null | xargs -r dirname)
endif
ifneq ($(strip $(CUDA_DRIVER_LIB_PATH)),)
    INFO_MSG_CUDA_DRIVER := "CUDA Driver:   Found system library -> $(CUDA_DRIVER_LIB_PATH)"
else
    INFO_MSG_CUDA_DRIVER := "CUDA Driver:   Warning! Could not find libcuda.so automatically."
endif

# --- Print Configuration ---
$(info =================================================================)
$(info Configuration:)
$(info $(INFO_MSG_CUDA_TOOLKIT))
$(info $(INFO_MSG_CUDA_DRIVER))
$(info =================================================================)

endif

# --- Configure Compiler and Linker with Paths ---
CXXFLAGS += -I$(HETEROSTA_INC_DIR) -I$(COMMON_DIR)
LDFLAGS  += -L$(HETEROSTA_LIB_DIR)
LDFLAGS  += -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -L$(CUDA_DRIVER_LIB_PATH)
LDFLAGS  += -Wl,-rpath,$(HETEROSTA_LIB_DIR) -Wl,-rpath,$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib

# --- Build Rules ---
.PHONY: all clean

all: $(TARGETS)

# Rule for read_netlist_demo
read_netlist_demo: read_netlist_demo.o
	@echo "[LD] Linking $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Rule for multi_gpu_demo
multi_gpu_demo: multi_gpu_demo.o
	@echo "[LD] Linking $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Rule for set_netlistdb_demo
set_netlistdb_demo: set_netlistdb_demo.o placedb.o
	@echo "[LD] Linking $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

placedb.o: placedb.cpp
	@echo "[CC] Compiling $< from common/"
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp
	@echo "[CC] Compiling $<"
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	@echo "[RM] Cleaning up..."
	$(RM) $(CLEAN_FILES)