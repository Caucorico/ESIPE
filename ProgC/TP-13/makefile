LDFLAGS=$(shell pkg-config --libs-only-l MLV)
CFLAGS=-W -Wall -ansi -pedantic

main: main.o bitboard.o graph_board.o
	gcc main.o bitboard.o graph_board.o -o main $(LDFLAGS)

main.o: main.c bitboard.h graph_board.h
	gcc -c main.c -o main.o $(CFLAGS)

bitboard.o: bitboard.c bitboard.h
	gcc -c bitboard.c -o bitboard.o $(CFLAGS)

graph_board.o: graph_board.c graph_board.h
	gcc -c graph_board.c -o graph_board.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o