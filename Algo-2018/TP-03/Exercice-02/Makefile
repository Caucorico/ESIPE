CC=gcc
CFLAGS=-Wall -ansi
LDFLAGS=$(shell pkg-config --libs-only-l MLV)
DEPS=grid.h draw.h
OBJ=sweeper.o grid.o draw.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

sweeper: $(OBJ)
	gcc -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) sweeper