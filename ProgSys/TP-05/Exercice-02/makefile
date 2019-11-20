CFLAGS=-Wall -pedantic -W -ansi
LDFLAGS=

main: main.o dispatch.o calc.o calc_entry.o
	gcc main.o dispatch.o calc.o -o main $(LDFLAGS)

main.o: main.c dispatch.h calc.h dispatch.h
	gcc -c main.c -o main.o $(CFLAGS)

calc_entry.o: calc_entry.c calc.h dispatch.h
	gcc -c calc_entry.c -o calc_entry.o $(CFLAGS)

calc.o: calc.c calc.h dispatch.h
	gcc -c calc.c -o calc.o $(CFLAGS)

dispatch.o: dispatch.c dispatch.h
	gcc -c dispatch.c -o dispatch.o $(CFLAGS)

run:
	./main

clear:
	rm -f *.o
	rm -f main
