# ======================================
# Makefile for the HeteroSTA C++ Demos
# ======================================

# --- Compiler and Targets ---
CXX := g++
RM  := rm -f
TARGETS := read_netlist_demo set_netlistdb_demo

# --- Source and Object Files ---
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
EXTRA_OBJECTS := placedb.o
CLEAN_FILES := $(TARGETS) $(OBJECTS) $(EXTRA_OBJECTS)

# --- Flags ---
CXXFLAGS := -std=c++17 -g -Wall -fPIC
LDFLAGS  :=
LDLIBS   := -lheterosta.cpu -lgomp -lrt -pthread -ldl

# --- Paths (USER CONFIGURABLE for HeteroSTA) ---
HETEROSTA_ROOT ?= ../..
HETEROSTA_INC_DIR ?= $(HETEROSTA_ROOT)/include
HETEROSTA_LIB_DIR ?= $(HETEROSTA_ROOT)/lib
COMMON_DIR := ../common
vpath %.cpp $(COMMON_DIR)

# --- Configure Compiler and Linker with Paths ---
CXXFLAGS += -I$(HETEROSTA_INC_DIR) -I$(COMMON_DIR)
LDFLAGS  += -L$(HETEROSTA_LIB_DIR)
LDFLAGS  += -Wl,-rpath,$(HETEROSTA_LIB_DIR)

# --- Build Rules ---
.PHONY: all clean

all: $(TARGETS)

# Rule for read_netlist_demo
read_netlist_demo: read_netlist_demo.o
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