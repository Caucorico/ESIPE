LDFLAGS=$(shell pkg-config --libs-only-l MLV)
CFLAGS=-W -Wall -ansi -pedantic

main: main.o bitboard.o graph_board.o board.o event.o
	gcc main.o bitboard.o graph_board.o board.o event.o -o main $(LDFLAGS)

main.o: main.c bitboard.h graph_board.h board.h event.h
	gcc -c main.c -o main.o $(CFLAGS)

bitboard.o: bitboard.c bitboard.h
	gcc -c bitboard.c -o bitboard.o $(CFLAGS)

graph_board.o: graph_board.c graph_board.h board.h
	gcc -c graph_board.c -o graph_board.o $(CFLAGS)

event.o: event.c event.h board.h bitboard.h
	gcc -c event.c -o event.o $(CFLAGS)

board.o: board.c board.h bitboard.h
	gcc -c board.c -o board.o $(CFLAGS)

clear:
	rm -f main
	rm -f *.o

run:
	./main