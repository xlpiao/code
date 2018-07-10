FLAGS=
DEBUG=-Wall -O0 -g

all: malloc matrix 

malloc: malloc.c
		gcc -o bin.malloc malloc.c $(FLAGS) $(DEBUG)

matrix: matrix.c
		gcc -o bin.matrix matrix.c $(FLAGS) $(DEBUG)

clean:
		rm bin.*
