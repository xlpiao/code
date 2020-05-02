FLAGS=
DEBUG=-Wall -O0 -g

all: malloc matrix scanf bubble_sort merge_sort dfs

malloc: malloc.c
	gcc -o bin.malloc malloc.c $(FLAGS) $(DEBUG)

matrix: matrix.c
	gcc -o bin.matrix matrix.c $(FLAGS) $(DEBUG)

bubble_sort: bubble_sort.c
	gcc -o bin.bubble_sort bubble_sort.c $(FLAGS) $(DEBUG)

merge_sort: merge_sort.c
	gcc -o bin.merge_sort merge_sort.c $(FLAGS) $(DEBUG)

dfs: dfs.cpp
	clang++ -o bin.dfs dfs.cpp -std=c++14 $(FLAGS) $(DEBUG)

scanf: scanf.c
	gcc -o bin.scanf scanf.c $(FLAGS) $(DEBUG)
	./bin.scanf < data.txt

clean:
	rm bin.*
