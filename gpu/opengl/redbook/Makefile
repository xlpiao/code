OS=$(shell uname -s)

ifeq ($(OS), Linux)
	LIBS=-lGL -lGLU -lglut
else
	LIBPATH=-L/System/Library/Frameworks
	LIBS=-framework GLUT -framework OpenGL -framework Cocoa
	# LIBS=-framework GLUT -framework OpenGL -framework Cocoa
endif

all:  aaindex aapoly aargb accanti accpersp alpha alpha3D bezcurve bezmesh bezsurf clip colormat cube dof double drawf feedback fog fogindex font hello image light lines list material model movelight pickdepth picksquare planet polys quadric robot scene select smooth stencil stroke surface teapots tess tesswind checker mipmap polyoff texbind texgen texprox texsub varray wrap texturesurf torus trim unproject


aaindex: aaindex.c
	clang -o bin.aaindex aaindex.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
aapoly: aapoly.c
	clang -o bin.aapoly aapoly.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
aargb: aargb.c
	clang -o bin.aargb aargb.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
accanti: accanti.c
	clang -o bin.accanti accanti.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
accpersp: accpersp.c
	clang -o bin.accpersp accpersp.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
alpha: alpha.c
	clang -o bin.alpha alpha.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
alpha3D: alpha3D.c
	clang -o bin.alpha3D alpha3D.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
bezcurve: bezcurve.c
	clang -o bin.bezcurve bezcurve.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
bezmesh: bezmesh.c
	clang -o bin.bezmesh bezmesh.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
bezsurf: bezsurf.c
	clang -o bin.bezsurf bezsurf.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
clip: clip.c
	clang -o bin.clip clip.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
colormat: colormat.c
	clang -o bin.colormat colormat.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
cube: cube.c
	clang -o bin.cube cube.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
dof: dof.c
	clang -o bin.dof dof.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
double: double.c
	clang -o bin.double double.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
drawf: drawf.c
	clang -o bin.drawf drawf.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
feedback: feedback.c
	clang -o bin.feedback feedback.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
fog: fog.c
	clang -o bin.fog fog.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
fogindex: fogindex.c
	clang -o bin.fogindex fogindex.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
font: font.c
	clang -o bin.font font.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
hello: hello.c
	clang -o bin.hello hello.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
image: image.c
	clang -o bin.image image.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
light: light.c
	clang -o bin.light light.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
lines: lines.c
	clang -o bin.lines lines.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
list: list.c
	clang -o bin.list list.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
material: material.c
	clang -o bin.material material.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
model: model.c
	clang -o bin.model model.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
movelight: movelight.c
	clang -o bin.movelight movelight.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
pickdepth: pickdepth.c
	clang -o bin.pickdepth pickdepth.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
picksquare: picksquare.c
	clang -o bin.picksquare picksquare.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
planet: planet.c
	clang -o bin.planet planet.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
polys: polys.c
	clang -o bin.polys polys.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
quadric: quadric.c
	clang -o bin.quadric quadric.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
robot: robot.c
	clang -o bin.robot robot.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
scene: scene.c
	clang -o bin.scene scene.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
select: select.c
	clang -o bin.select select.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
smooth: smooth.c
	clang -o bin.smooth smooth.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
stencil: stencil.c
	clang -o bin.stencil stencil.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
stroke: stroke.c
	clang -o bin.stroke stroke.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
surface: surface.c
	clang -o bin.surface surface.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
teapots: teapots.c
	clang -o bin.teapots teapots.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
tess: tess.c
	clang -o bin.tess tess.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
tesswind: tesswind.c
	clang -o bin.tesswind tesswind.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
checker: checker.c
	clang -o bin.checker checker.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
mipmap: mipmap.c
	clang -o bin.mipmap mipmap.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
polyoff: polyoff.c
	clang -o bin.polyoff polyoff.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
texbind: texbind.c
	clang -o bin.texbind texbind.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
texgen: texgen.c
	clang -o bin.texgen texgen.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
texprox: texprox.c
	clang -o bin.texprox texprox.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
texsub: texsub.c
	clang -o bin.texsub texsub.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
varray: varray.c
	clang -o bin.varray varray.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
wrap: wrap.c
	clang -o bin.wrap wrap.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
texturesurf: texturesurf.c
	clang -o bin.texturesurf texturesurf.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
torus: torus.c
	clang -o bin.torus torus.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
trim: trim.c
	clang -o bin.trim trim.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations
unproject: unproject.c
	clang -o bin.unproject unproject.c $(LIBPATH) $(LIBS) -Wno-deprecated-declarations


clean:
	rm -f bin.aaindex bin.aapoly bin.aargb bin.accanti bin.accpersp bin.alpha bin.alpha3D bin.bezcurve bin.bezmesh bin.bezsurf bin.clip bin.colormat bin.cube bin.dof bin.double bin.drawf bin.feedback bin.fog bin.fogindex bin.font bin.hello bin.image bin.light bin.lines bin.list bin.material bin.model bin.movelight bin.pickdepth bin.picksquare bin.planet bin.polys bin.quadric bin.robot bin.scene bin.select bin.smooth bin.stencil bin.stroke bin.surface bin.teapots bin.tess bin.tesswind bin.checker bin.mipmap bin.polyoff bin.texbind bin.texgen bin.texprox bin.texsub bin.varray bin.wrap bin.texturesurf bin.torus bin.trim bin.unproject
