FLAGS=
DEBUG=-Wall -O0 -g

all: malloc matrix scanf

malloc: malloc.c
		gcc -o bin.malloc malloc.c $(FLAGS) $(DEBUG)

matrix: matrix.c
		gcc -o bin.matrix matrix.c $(FLAGS) $(DEBUG)

scanf: scanf.c
		gcc -o bin.scanf scanf.c $(FLAGS) $(DEBUG)
		./bin.scanf < data.txt

clean:
		rm bin.*
