from setup import *
from mpi4py.MPI import COMM_WORLD as comm
 
spy = SPY(params, datapath, 'baseflow', direction_map)
spy.loadBaseflow(Re,S)
ud,_=spy.Q.split()

with dfx.io.XDMFFile(comm, "line.xdmf", "r") as file: mesh = file.read_mesh()

cell = ufl.Cell("interval", geometric_dimension=2)
VE = ufl.VectorElement("Lagrange", cell, 1, 3)
u = Function(dfx.fem.FunctionSpace(mesh, VE))
u.interpolate(ud)

with dfx.io.XDMFFile(comm, "export.xdmf", "w") as xdmf:
	xdmf.write_mesh(mesh)
	xdmf.write_function(u)