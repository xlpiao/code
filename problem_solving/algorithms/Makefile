FLAGS=--std=c++11
DEBUG=-Wall -O0 -g

all: malloc matrix scanf Sort merge_sort dfs

malloc: malloc.cc
	gcc -o bin.malloc malloc.cc $(FLAGS) $(DEBUG)

matrix: matrix.cc
	gcc -o bin.matrix matrix.cc $(FLAGS) $(DEBUG)

Sort: Sort.cc
	clang++ -o bin.Sort Sort.cc $(FLAGS) $(DEBUG)

merge_sort: merge_sort.c
	gcc -o bin.merge_sort merge_sort.c $(FLAGS) $(DEBUG)

dfs: dfs.cpp
	clang++ -o bin.dfs dfs.cpp -std=c++14 $(FLAGS) $(DEBUG)

scanf: scanf.c
	gcc -o bin.scanf scanf.c $(FLAGS) $(DEBUG)
	./bin.scanf < data.txt

clean:
	rm bin.*
