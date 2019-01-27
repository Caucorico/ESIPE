main: stack_height.o
	gcc stack_height.o -o main -Wall -ansi -pedantic -W

stack_height.o: stack_height.c
	gcc -c stack_height.c -o stack_height.o -Wall -ansi -pedantic -W

clear:
	rm -f main
	rm -f *.o

run:
	./main