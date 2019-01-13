test: main.o ex1.o ex2.o ex3.o ex4.o ex5.o ex6.o
	gcc -o $@ $^

%.o: %.c
	$(CC) -o $@ -c $<

clean:
	rm -f *.o
	rm -f test

run:
	./test
