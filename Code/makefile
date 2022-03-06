CC = gcc
PROGRAM_NAME = main
OBJECTS = main.o Lab3IO.o
CLIENT_NAME = client

$(PROGRAM_NAME): main.o Lab3IO.o
	$(CC) -fopenmp -o $(PROGRAM_NAME) $(OBJECTS) -lm

main.o: main.c Lab3IO.h timer.h
	$(CC) -fopenmp -c main.c -o main.o

Lab3IO.o: Lab3IO.c
	$(CC) -fopenmp -c Lab3IO.c -o Lab3IO.o

clean:
	rm *.o $(PROGRAM_NAME)
