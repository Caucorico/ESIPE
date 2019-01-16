CC=gcc
CFLAGS=-Wall -ansi
LDFLAGS=$(shell pkg-config --libs-only-l MLV)
DEPS=visualarray.h array.h sort.h
OBJ=array.o sort.o
OBJMLV=visualarray.o array.o sort.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

tp04: tp04.o $(OBJ)
	gcc -o $@ $^

demo1: demo1.o $(OBJMLV)
	gcc -o $@ $^ $(LDFLAGS)

demo2: demo2.o $(OBJMLV)
	gcc -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o tp04 demo1 demo2