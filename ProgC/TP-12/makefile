CFLAGS=-Wall -ansi -pedantic -W
LDFLAGS=

main: main.o word_list.o parser.o hash.o
	gcc main.o word_list.o parser.o hash.o -o main $(LDFLAGS)

main.o: main.c parser.h word_list.h hash.h
	gcc -c main.c -o main.o $(CFLAGS)

parser.o: parser.c parser.h word_list.h hash.h
	gcc -c parser.c -o parser.o $(CFLAGS)

word_list.o: word_list.c word_list.h hash.h
	gcc -c word_list.c -o word_list.o $(CFLAGS)

hash.o: hash.c hash.h word_list.h
	gcc -c hash.c -o hash.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o

run:
	./main