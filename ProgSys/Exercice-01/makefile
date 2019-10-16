CFLAGS=-ansi -Wall -W -pedantic
LDFLAGS=

stat: main.o stat.o
	gcc -o stat main.o stat.o

main.o: main.c stat.h
	gcc -o main.o -c main.c $(CFLAGS)

stat.o: stat.c stat.h
	gcc -o stat.o -c stat.c $(CFLAGS)

clear:
	rm -f *.o
	rm -f stat

run:
	./stat
