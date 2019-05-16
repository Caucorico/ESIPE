LDFLAGS=
CFLAGS=-W -Wall -ansi -pedantic

main: main.o bitboard.o
	gcc main.o bitboard.o -o main $(LDFLAGS)

main.o: main.c bitboard.h
	gcc -c main.c -o main.o $(CFLAGS)

bitboard.o: bitboard.c bitboard.h
	gcc -c bitboard.c -o bitboard.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o