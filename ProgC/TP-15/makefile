LDFLAGS=-lreadline
CFLAGS=-Wall -ansi -pedantic -W

main: main.o stack.o pol_calc.o
	gcc main.o stack.o pol_calc.o -o main $(LDFLAGS)

main.o: main.c pol_calc.h
	gcc -c main.c -o main.o $(CFLAGS)

pol_caclc.o: pol_calc.c pol_calc.h stack.h
	gcc -c pol_calc.c -o pol_calc.o $(CFLAGS)

stack.o: stack.c stack.h
	gcc -c stack.c -o stack.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o