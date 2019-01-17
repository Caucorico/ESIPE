CC=gcc
CFLAGS=-Wall -ansi
DEPS=array.h sort.h
OBJ=array.o sort.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: main.o $(OBJ)
	gcc -o $@ $^

.PHONY: clean
clean:
	rm -f *.o main