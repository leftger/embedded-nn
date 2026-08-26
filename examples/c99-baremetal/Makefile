CC ?= gcc
CFLAGS ?= -std=c99 -Wall -Wextra -O3 -Iinclude

TARGET = c99_baremetal_inference
SRCS = src/main.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
