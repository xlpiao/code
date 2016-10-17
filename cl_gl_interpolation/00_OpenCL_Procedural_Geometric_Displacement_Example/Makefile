OS=$(shell uname -s)

ifeq ($(OS), Linux)
	LIBS=-lGL -lGLU -lglut -lOpenCL -lpthread -lm -std=gnu99
else
	LIBPATH=-L/System/Library/Frameworks
	LIBS=-framework GLUT -framework OpenGL -framework Cocoa -framework opencl -stdlib=libc++ -lstdc++
	# LIBS=-framework GLUT -framework OpenGL -framework Cocoa
endif

  all: displacement

displacement: displacement.c
	clang -o bin.displacement displacement.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

clean:
	rm -f bin.displacement
