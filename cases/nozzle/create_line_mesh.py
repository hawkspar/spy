import ufl
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_mesh
from mpi4py.MPI import COMM_WORLD as comm

cell = ufl.Cell("interval", geometric_dimension=2)
VE = ufl.VectorElement("Lagrange", cell, 1, 3)
domain = ufl.Mesh(VE)

n=1000
x = np.vstack((np.ones(n),np.linspace(0,2,n))).T
L = np.arange(n, dtype=np.int64)
cells = np.vstack((L[:-1],L[1:])).T

mesh = create_mesh(comm, cells, x, domain)

with XDMFFile(comm, "line.xdmf", "w") as xdmf: xdmf.write_mesh(mesh)