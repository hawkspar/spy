import ufl
import numpy as np
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py.MPI import COMM_WORLD
import meshio

# Tutorial extract
mesh = meshio.read("Mesh/validation/validation.msh")
cells = mesh.get_cells_type("triangle")
cell_data = mesh.get_cell_data("gmsh:physical", "triangle")
mesh = meshio.Mesh(points=mesh.points[:,:2], cells={"triangle": cells}, cell_data={"name_to_read":[cell_data]})
meshio.write("mesh.xdmf", mesh)

# Reread directly xdmf file
with XDMFFile(COMM_WORLD, "mesh.xdmf", "r") as file:
	mesh = file.read_mesh(name="Grid")

# Create Finite Element and test for outliers
def test_outliers():
	FE=ufl.FiniteElement("Lagrange",mesh.ufl_cell(),2)
	FS=dfx.FunctionSpace(mesh,FE)
	fx=dfx.Function(FS)
	fx.interpolate(lambda x: x[0])
	ux=fx.compute_point_values()
	print(np.sum(ux<0))
	print(ux[ux<0])

test_outliers()
# Attempt to nudge outliers into place
x = mesh.geometry.x
x[x[:,0]<0,0] = 0
test_outliers()