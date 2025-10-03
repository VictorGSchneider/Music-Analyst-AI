CC = mpicc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -pedantic
LDFLAGS ?=
TARGET := mpi_spotify
SRC := src/mpi_spotify.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
