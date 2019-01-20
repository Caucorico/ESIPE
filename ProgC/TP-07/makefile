CC=gcc
CFLAGS=-Wall -ansi -lMLV
LDFLAGS=-lMLV
OBJ=sudoku.o main.o in_out.o graph.o

sudoku: $(OBJ)
	$(CC) -o sudoku $(OBJ) $(CFLAGS)

graph.o: graph.c graph.h
	$(CC) -c graph.c $(CFLAGS)

main.o: main.c sudoku.h in_out.h graph.h
	$(CC) -c main.c $(CFLAGS)

sudoku.o: sudoku.c sudoku.h
	$(CC) -c sudoku.c $(CFLAGS)

in_out.o: in_out.c in_out.h
	$(CC) -c in_out.c $(CFLAGS)

clean: 
	rm -f *.o
	rm -f *~
	rm -f sudoku
