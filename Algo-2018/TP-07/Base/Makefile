CC=gcc
CFLAGS=-Wall -ansi
LDFLAGS=-lm
DEPS=list.h table.h
OBJ=list.o table.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

concordance: concordance.c $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) concordance