CFLAGS=-ansi -Wall -W -pedantic
LDFLAGS=

stat: main.o stat.o dirstat.o
	gcc -o stat main.o stat.o dirstat.o

main.o: main.c stat.h dirstat.h
	gcc -o main.o -c main.c $(CFLAGS)

dirstat.o: dirstat.c dirstat.h stat.h
	gcc -o dirstat.o -c dirstat.c $(CFLAGS)

stat.o: stat.c stat.h
	gcc -o stat.o -c stat.c $(CFLAGS)

clear:
	rm -f *.o
	rm -f stat

run:
	./stat
