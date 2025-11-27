CXX = riscv64-unknown-linux-gnu-g++
CC = riscv64-unknown-linux-gnu-gcc
SYSROOT_DIR ?= ../../../../../sw/firesim-software/boards/firechip/distros/br/buildroot/output/host/usr/riscv64-buildroot-linux-gnu/sysroot

CXXFLAGS = \
	-levent \
	-lzmq \
	-lrt \
	--sysroot=$(SYSROOT_DIR) \
	-pthread \
	-O3 \
	-Wall \
	-g \
	-std=c++11 \
	-D_GNU_SOURCE

all: optstuff
	$(CXX) $(CXXFLAGS) mutilate/mutilate.cc cmdline.cc mutilate/log.cc mutilate/distributions.cc mutilate/util.cc mutilate/Connection.cc mutilate/Protocol.cc mutilate/Generator.cc mutilate/barrier.cc -o mutilate3

optstuff:
	gengetopt < mutilate/cmdline.ggo

.PHONY: optstuff all
