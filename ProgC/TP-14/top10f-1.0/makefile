LDFLAGS=
CFLAGS=-Wall -ansi -pedantic -W

top10f: main.o top10f.o
	gcc main.o top10f.o -o main $(LDFLAGS)

main.o: main.c top10f.h
	gcc -c main.c -o main.o $(CFLAGS)

top10f.o: top10f.c top10f.h
	gcc -c top10f.c -o top10f.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o