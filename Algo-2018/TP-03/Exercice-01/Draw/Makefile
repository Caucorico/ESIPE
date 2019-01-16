CC=gcc
CFLAGS=-Wall -ansi
LDFLAGS=$(shell pkg-config --libs-only-l MLV)
DEPS=
OBJ=draw.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

draw: draw.o
	gcc -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) draw