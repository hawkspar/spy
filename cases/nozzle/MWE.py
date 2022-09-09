import ufl
import PetscBinaryIO
import dolfinx as dfx
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function

with dfx.io.XDMFFile(comm, "nozzle.xdmf", "r") as file:
	mesh = file.read_mesh(name="Grid")

# file handler and complex mode
io = PetscBinaryIO.PetscBinaryIO(complexscalars=True)

# Finite elements & function spaces
FE_vector=ufl.VectorElement("CG",mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("CG",mesh.ufl_cell(),1)
V = FunctionSpace(mesh,FE_vector)
W = FunctionSpace(mesh,FE_scalar)
TH=FunctionSpace(mesh,FE_vector*FE_scalar)
U  =Function(V)
P  =Function(W)
nut=Function(W)

u_in   = io.readBinaryFile(f"./baseflow/u/complex/u_S=0.000_Re=400000_n=10_p={comm.rank:d}.dat")[0]
p_in   = io.readBinaryFile(f"./baseflow/p/complex/p_S=0.000_Re=400000_n=10_p={comm.rank:d}.dat")[0]
nut_in = io.readBinaryFile(f"./baseflow/nut/complex/nut_S=0.000_Re=400000_n=10_p={comm.rank:d}.dat")[0]
U.vector[...] = u_in
P.vector[...] = p_in
nut.vector[...] = nut_in
U.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
P.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
nut.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)

with dfx.io.XDMFFile(comm, "sanity_check_U_MWE.xdmf", "w") as xdmf:
	xdmf.write_mesh(mesh)
	xdmf.write_function(U)
with dfx.io.XDMFFile(comm, "sanity_check_P_MWE.xdmf", "w") as xdmf:
	xdmf.write_mesh(mesh)
	xdmf.write_function(P)
with dfx.io.XDMFFile(comm, "sanity_check_nut_MWE.xdmf", "w") as xdmf:
	xdmf.write_mesh(mesh)
	xdmf.write_function(nut)