from setup import *
from mpi4py.MPI import COMM_WORLD as comm
 
S=1
spy = SPY(params, data_path, 'baseflow', direction_map)
spy.loadBaseflow(Re,S)
ud,pd=spy.Q.split()

with dfx.io.XDMFFile(comm, "line.xdmf", "r") as file: mesh = file.read_mesh()

cell = ufl.Cell("interval", geometric_dimension=2)
V = ufl.VectorElement("Lagrange", cell, 2, 3)
F = ufl.FiniteElement("Lagrange", cell, 1)
u = Function(dfx.fem.FunctionSpace(mesh, V))
p = Function(dfx.fem.FunctionSpace(mesh, F))
u.interpolate(ud)
p.interpolate(pd)

with dfx.io.XDMFFile(comm, "export_u.xdmf", "w") as xdmf:
	xdmf.write_mesh(mesh)
	xdmf.write_function(u)

with dfx.io.XDMFFile(comm, "export_p.xdmf", "w") as xdmf:
	xdmf.write_mesh(mesh)
	xdmf.write_function(p)