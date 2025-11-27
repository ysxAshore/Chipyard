#################################
# RISCV Toolchain
#################################

PREFIX = /opt/autovec-baremetal-rvv/bin/riscv64-unknown-elf-

GCC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size

sysroot = $(shell $(GCC) -print-sysroot)
# GCC = clang
# sysroot = /opt/rv64gcv_lp64d_enable_llvm/riscv64-unknown-elf
#################################
# Flags
#################################

# SoC Settings
ARCH = rv64imafdcv
ABI = lp64d
ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI)

CFLAGS  = -std=gnu99 -g -fno-common -fno-builtin-printf -fpermissive -Wall -O3
CFLAGS += $(ARCHFLAGS)
LDFLAGS = -static

# include libgloss.mk

libgloss_specs := htif_nano.specs
CFLAGS += -specs=$(libgloss_specs)
LDFLAGS += -specs=$(libgloss_specs)


# PROGRAMS = pwm blkdev accum charcount nic-loopback big-blkdev pingd \
        #    streaming-passthrough streaming-fir nvdla spiflashread spiflashwrite fft gcd \
        #    hello mt-hello 8
# PROGRAMS = conv_MMA_padding_scalar_10  conv_MMA_padding_scalar_23 conv_MMA_padding_scalar_11  conv_MMA_padding_scalar_3 conv_MMA_padding_scalar_12  conv_MMA_padding_scalar_4 conv_MMA_padding_scalar_13  conv_MMA_padding_scalar_5 conv_MMA_padding_scalar_14  conv_MMA_padding_scalar_6 conv_MMA_padding_scalar_15  conv_MMA_padding_scalar_7 conv_MMA_padding_scalar_16  conv_MMA_padding_scalar_8 conv_MMA_padding_scalar_17  conv_MMA_padding_scalar_9 conv_MMA_padding_scalar_18  conv_MMA_padding_scalar_19  conv_MMA_padding_scalar_2   conv_MMA_padding_scalar_20  conv_MMA_padding_scalar_21  conv_MMA_padding_scalar_1     conv_MMA_padding_scalar_22
PROGRAMS = resnet50_cpu_int8_CUTE_chipyard_run_pure vec_ops_only_2 vec_ops_only_3 vec_ops_only_4 vec_ops_only_5 vec_ops_only_6 vec_ops_only_7 vec_ops_only_8 vec_ops_only_9 vec_ops_only_10 vec_ops_only_11 vec_ops_only_12 vec_ops_only_13 vec_ops_only_14 vec_ops_only_15 vec_ops_only_16 vec_ops_only_17 vec_ops_only_18 vec_ops_only_19 vec_ops_only_20 vec_ops_only_21 vec_ops_only_22 vec_ops_only_23 vec_ops_only_24 vec_ops_only_25 vec_ops_only_26 vec_ops_only_27 vec_ops_only_28 vec_ops_only_29 vec_ops_only_30 vec_ops_only_31 vec_ops_only_32 vec_ops_only_33 vec_ops_only_34 vec_ops_only_35 vec_ops_only_36 vec_ops_only_37 vec_ops_only_38 vec_ops_only_39 vec_ops_only_40 vec_ops_only_41 vec_ops_only_42 vec_ops_only_43 vec_ops_only_44 vec_ops_only_45 vec_ops_only_46 vec_ops_only_47 vec_ops_only_48 vec_ops_only_49 vec_ops_only_50 vec_ops_only_51 vec_ops_only_52 vec_ops_only_53

.DEFAULT_GOAL := default


#################################
# Build
#################################


%.o: %.S
	$(GCC) $(CFLAGS) -D__ASSEMBLY__=1 -c $< -o $@

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

%.riscv: %.o $(libgloss)
	$(GCC) $(LDFLAGS) $< -o $@

%.dump: %.riscv
	$(OBJDUMP) -D -S $< > $@

# gcc get_test.c -o get_value
get_matrix_value: get_matrix_test.c
	$(CC) -o $@ $^

get_conv_value: get_conv_test.c
	$(CC) -o $@ $^

#################################
# Recipes
#################################

.PHONY: clean
clean:
	rm -f *.riscv *.o *.dump *.S
	$(if $(libgloss),rm -rf $(libgloss_builddir)/)

.PHONY: default
default: $(addsuffix .riscv, $(PROGRAMS))

.PHONY: dumps
dumps: $(addsuffix .dump, $(PROGRAMS))
