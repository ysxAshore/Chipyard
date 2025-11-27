# Compiler and flags
CC = gcc
CFLAGS = -fpermissive -Ofast -w
LDFLAGS = -lm

# Target executable
TARGET = transformer_cute

# Source file
SRC = transformer_cute_x86.c

# Rule to create the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean up the generated files
clean:
	rm -f $(TARGET)

.PHONY: all clean