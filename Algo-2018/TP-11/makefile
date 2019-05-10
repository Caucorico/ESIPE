CFLAGS=-W -Wall -ansi -pedantic
LDFLAGS=

main: main.o heap.o
	gcc main.o heap.o -o main $(LDFLAGS)

main.o: main.c heap.h
	gcc -c main.c -o main.o $(CFLAGS)

heap.o: heap.c heap.h
	gcc -c heap.c -o heap.o $(CFLAGS)

run:
	./main

clean:
	rm -f main
	rm -f *.o