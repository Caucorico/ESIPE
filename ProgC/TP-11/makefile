CFLAGS=-Wall -ansi -pedantic -W
LDFLAGS=$(shell pkg-config --libs-only-l MLV)

main: main.o board.o event.o window.o
	gcc main.o board.o event.o window.o -o main $(LDFLAGS)

main.o: main.c board.h window.h event.h
	gcc -c main.c -o main.o $(CFLAGS)

board.o: board.c board.h
	gcc -c board.c -o board.o $(CFLAGS)

event.o: event.c event.h board.h
	gcc -c event.c -o event.o $(CFLAGS)

window.o: window.c window.h board.h
	gcc -c window.c -o window.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o

run:
	./main