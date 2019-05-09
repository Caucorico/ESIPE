CFLAGS=-Wall -ansi -pedantic -W
LDFLAGS=

main: main.o word_list.o parser.o
	gcc main.o word_list.o parser.o -o main $(LDFLAGS)

main.o: main.c parser.h word_list.h
	gcc -c main.c -o main.o $(CFLAGS)

parser.o: parser.c parser.h word_list.h
	gcc -c parser.c -o parser.o $(CFLAGS)

word_list.o: word_list.c word_list.h
	gcc -c word_list.c -o word_list.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o

run:
	./main