CC=gcc
CFLAGS=-Wall -ansi
DEPS=arrays.h
OBJ=bench_main.o benchmark.o arrays_ref.o arrays.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

bench: $(OBJ)
	gcc -o $@ $^

.PHONY: clean
clean:
	rm -f bench_main.o benchmark.o arrays.o bench