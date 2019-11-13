CFLAGS=-Wall -pedantic -W -ansi
LDFLAGS=

main: main.o dispatch.o calc.o
	gcc main.o dispatch.o calc.o -o main $(LDFLAGS)

main.o: main.c dispatch.h calc.h
	gcc -c main.c -o main.o $(CFLAGS)

calc.o: calc.c calc.h dispatch.h
	gcc -c calc.c -o calc.o $(CFLAGS)

dispatch.o: dispatch.c dispatch.h
	gcc -c dispatch.c -o dispatch.h $(CFLAGS)

run:
	./main

clear:
	rm -f *.o
	rm -f main
