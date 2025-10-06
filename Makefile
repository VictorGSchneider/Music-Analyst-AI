MPICC ?= mpicc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic

SRC := src/parallel_spotify.c
BIN := bin/parallel_spotify

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -o $@ $^

clean:
	rm -rf bin output
