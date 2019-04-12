CC=gcc
CFLAGS=-Wall -ansi
LDFLAGS=
DEPS=visualtree.h tree.h
OBJ=visualtree.o tree.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

tp08: tp08.o $(OBJ)
	gcc -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) tp08 tp08.o current-tree.dot current-tree.pdf