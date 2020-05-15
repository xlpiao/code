CC=clang++
CFLAGS=`pkg-config --cflags pango pangoft2`
LIBS=`pkg-config --libs pango pangoft2`
DEBUG=-Wall -O0 -g

all: pangoft2

pangoft2: pangoft2.cc
	$(CC) pangoft2.cc -o bin.pangoft2 $(CFLAGS) $(LIBS) $(DEBUG)
	./bin.pangoft2

clean:
	rm -f bin.*
