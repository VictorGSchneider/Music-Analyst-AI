CC=mpicc
CFLAGS=-O2 -Wall -Wextra -std=c11 -Ithird_party
LDFLAGS=
TARGET=spotify_mpi
SRC=src/spotify_mpi.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
