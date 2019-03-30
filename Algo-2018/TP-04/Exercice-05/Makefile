CC=gcc
CFLAGS=-Wall -ansi
DEPS=array.h sort.h
OBJ=array.o sort.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

tp04-instr: tp04-instr.o $(OBJ)
	gcc -o $@ $^

.PHONY: clean
clean:
	rm -f *.o tp04-instr