main: main.o
	gcc main.o -o main -Wall -ansi -pedantic -W

main.o: main.c
	gcc -c main.c -o main.o -Wall -ansi -pedantic -W

clear:
	rm -f main
	rm -f *.o

run:
	./main