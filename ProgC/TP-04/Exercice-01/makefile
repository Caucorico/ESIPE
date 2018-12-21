main: main.o stack.o
	gcc -o main main.o stack.o

main.o: main.c stack.h
	gcc -c -o main.o main.c -Wall -ansi -pedantic -W

stack.o: stack.c
	gcc -c -o stack.o stack.c -Wall -ansi -pedantic -W

clean:
	rm -f *.o
	rm -f main