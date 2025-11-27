# Compiler settings
CC = gcc
CFLAGS = -fpermissive -w -Ofast

# Executable names
TARGETS = resnet50_dataflow resnet50_CUTE_Gloden_int8 get_transpose shift_int8 resnet50_gloden get_layer_1_out get_all_layer_out

# Source files
SRC1 = resnet50_cute_dataflow.c
SRC2 = resnet50_cpu_int8_CUTE_Gloden.c
SRC3 = resnet50_cpu_int8_quick_get_transpose.c
SRC4 = resnet50_cpu_int8_quick.c
SRC5 = resnet50_gloden.c
SRC6 = resnet50_cpu_int8_CUTE_Gloden_layer1_out.c
SRC7 = resnet50_cpu_int8_CUTE_Gloden_all_layer_out.c

# Executable rules
all: $(TARGETS)

resnet50_dataflow: $(SRC1)
	$(CC) $(CFLAGS) -o $@ $^

resnet50_CUTE_Gloden_int8: $(SRC2)
	$(CC) $(CFLAGS) -o $@ $^

get_transpose: $(SRC3)
	$(CC) $(CFLAGS) -o $@ $^

shift_int8: $(SRC4)
	$(CC) $(CFLAGS) -o $@ $^

resnet50_gloden: $(SRC5)
	$(CC) $(CFLAGS) -o $@ $^

get_layer_1_out: $(SRC6)
	$(CC) $(CFLAGS) -o $@ $^

get_all_layer_out: $(SRC7)
	$(CC) $(CFLAGS) -o $@ $^

# Run command for resnet50_CUTE_Gloden_int8
run_gloden: resnet50_CUTE_Gloden_int8
	./resnet50_CUTE_Gloden_int8 cpu conv check > CUTE_GLODEN.log

# Run command for get_transpose
run_transpose: get_transpose
	./get_transpose > transpose_w.h

# Run command for shift_int8
run_shift: shift_int8
	./shift_int8 cpu

run_gloden_cpu: resnet50_gloden
	./resnet50_gloden cpu conv > CPU_GLODEN.log

# Clean up the compiled files
clean:
	rm -f $(TARGETS) CUTE_GLODEN.log transpose_w.h CPU_GLODEN.log

.PHONY: all clean run_gloden run_transpose run_shift