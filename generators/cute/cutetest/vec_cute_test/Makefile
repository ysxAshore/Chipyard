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
PROGRAMS = resnet50_cpu_int8_CUTE_chipyard_run vec_ops_conv_2 vec_ops_conv_3 vec_ops_conv_4 vec_ops_conv_5 vec_ops_conv_6 vec_ops_conv_7 vec_ops_conv_8 vec_ops_conv_9 vec_ops_conv_10 vec_ops_conv_11 vec_ops_conv_12 vec_ops_conv_13 vec_ops_conv_14 vec_ops_conv_15 vec_ops_conv_16 vec_ops_conv_17 vec_ops_conv_18 vec_ops_conv_19 vec_ops_conv_20 vec_ops_conv_21 vec_ops_conv_22 vec_ops_conv_23 vec_ops_conv_24 vec_ops_conv_25 vec_ops_conv_26 vec_ops_conv_27 vec_ops_conv_28 vec_ops_conv_29 vec_ops_conv_30 vec_ops_conv_31 vec_ops_conv_32 vec_ops_conv_33 vec_ops_conv_34 vec_ops_conv_35 vec_ops_conv_36 vec_ops_conv_37 vec_ops_conv_38 vec_ops_conv_39 vec_ops_conv_40 vec_ops_conv_41 vec_ops_conv_42 vec_ops_conv_43 vec_ops_conv_44 vec_ops_conv_45 vec_ops_conv_46 vec_ops_conv_47 vec_ops_conv_48 vec_ops_conv_49 vec_ops_conv_50 vec_ops_conv_51 vec_ops_conv_52 vec_ops_conv_53

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
