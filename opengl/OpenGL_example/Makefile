OS=$(shell uname -s)

ifeq ($(OS), Linux)
	LIBS=-lGL -lGLU -lglut
else
	LIBPATH=-L/System/Library/Frameworks
	LIBS=-framework GLUT -framework OpenGL -framework Cocoa
	# LIBS=-framework GLUT -framework OpenGL -framework Cocoa
endif

  all: triangle rectangle SpinningCube VertexArray DisplayLists Viewport           RenderToTexture            Outlines       OpenGLInfo
# all: triangle rectangle SpinningCube VertexArray DisplayLists Viewport Billboard RenderToTexture TextureGen Outlines Water OpenGLInfo

triangle: triangle.c
	clang -o bin.triangle triangle.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

rectangle: rectangle.c
	clang -o bin.rectangle rectangle.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

SpinningCube: SpinningCube.c
	clang -o bin.SpinningCube SpinningCube.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

VertexArray: VertexArray.c
	clang -o bin.VertexArray VertexArray.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

DisplayLists: DisplayLists.c
	clang -o bin.DisplayLists DisplayLists.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

Viewport: Viewport.c
	clang -o bin.Viewport Viewport.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

# Billboard: Billboard.c
# 	clang -o bin.Billboard Billboard.c $(LIBPATH) $(LIBS) -framework JPEG -Wno-deprecated-declarations

RenderToTexture: RenderToTexture.c
	clang -o bin.RenderToTexture RenderToTexture.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

# TextureGen: TextureGen.c
# 	clang -o bin.TextureGen TextureGen.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

Outlines: Outlines.c
	clang -o bin.Outlines Outlines.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

# Water: Water.c
# 	clang -o bin.Water Water.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

OpenGLInfo: OpenGLInfo.c
	clang -o bin.OpenGLInfo OpenGLInfo.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

DumpState: DumpState.c
	clang -o bin.DumpState DumpState.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

GLInfo: GLInfo.c
	clang -o bin.GLInfo GLInfo.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations

clean:
	rm -f bin.triangle bin.rectangle bin.SpinningCube bin.VertexArray bin.DisplayLists bin.Viewport bin.Billboard bin.RenderToTexture bin.TextureGen bin.Outlines bin.Water bin.OpenGLInfo bin.DumpState bin.GLInfo
