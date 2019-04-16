main: main.o universe.o body_list.o body.o vector.o
	gcc main.o universe.o body_list.o body.o vector.o -o main -Wall -ansi -pedantic -W

main.o: main.c universe.h
	gcc -c main.c -o main.o -Wall -ansi -pedantic -W

universe.o: universe.c universe.h body_list.h
	gcc -c universe.c -o universe.o -Wall -ansi -pedantic -W

body_list.o: body_list.c body_list.h body.h
	gcc -c body_list.c -o body_list.o -Wall -ansi -pedantic -W

body.o: body.c body.h vector.h
	gcc -c body.c -o body.o -Wall -ansi -pedantic -W

vector.o: vector.c vector.h
	gcc -c vector.c -o vector.o -Wall -ansi -pedantic -W

clear:
	rm -f main
	rm -f *.o

run:
	./main
