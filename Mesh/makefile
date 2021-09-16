P=Mesh

all: $(P).xdmf

$(P).msh: $(P).geo
	gmsh -2 $<

$(P).xdmf: $(P).msh
	meshio-convert -pz $< $(P).xdmf

clean:
	rm -rf *.msh *.xdmf *.pvd *.vtu
