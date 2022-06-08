import meshio, ufl
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc as pet

p0=comm.rank==0

if p0:
    # Read mesh and point data
    openfoam_data = meshio.read("front.xmf")
    # Write it out again
    ps = openfoam_data.points[:,:2]
    cs = openfoam_data.get_cells_type("quad")
    dolfinx_fine_mesh = meshio.Mesh(points=ps, cells={"quad": cs})
    meshio.write("nozzle.xdmf", dolfinx_fine_mesh)
else: openfoam_data=None
openfoam_data = comm.bcast(openfoam_data, root=0) # openfoam_data available to all but not distributed
# Read it again in dolfinx - now it's a dolfinx object
with XDMFFile(comm, "nozzle.xdmf", "r") as file:
    dolfinx_fine_mesh = file.read_mesh(name="Grid")

# Create FiniteElement, FunctionSpace & Functions
FE_vector_1=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),1)
V1=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_1)
V2=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2)
W=dfx.FunctionSpace(dolfinx_fine_mesh,FE_scalar)
u1, u2 = dfx.Function(V1), dfx.Function(V2)
p = dfx.Function(W)

# Global indexes owned locally
ids = dolfinx_fine_mesh.geometry.input_global_indices
# Map OpenFOAM data directy onto dolfinx vectors
u1.x.array[:]  = openfoam_data.point_data['U'][  ids,:].flatten()
p.x.array[:]   = openfoam_data.point_data['p'][  ids]
# Interpolation to higher order
u2.interpolate(u1)
with XDMFFile(comm, "sanity_check_parallel.xdmf", "w") as xdmf:
    xdmf.write_mesh(dolfinx_fine_mesh)
    xdmf.write_function(u2)
# Write result as mixed
TH = dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2*FE_scalar)
_, dofs_U = TH.sub(0).collapse(collapsed_dofs=True)
_, dofs_p = TH.sub(1).collapse(collapsed_dofs=True)
q = dfx.Function(TH)
q.x.array[dofs_U]=u2.x.array
q.x.array[dofs_p]=p.x.array

viewer = pet.Viewer().createMPIIO("baseflow.dat", 'w', comm)
q.vector.view(viewer)
viewer = pet.Viewer().createMPIIO("baseflow.dat", 'r', comm)
q.vector.load(viewer)
u,p=q.split()
with XDMFFile(comm, "sanity_check_parallel2.xdmf", "w") as xdmf:
    xdmf.write_mesh(dolfinx_fine_mesh)
    xdmf.write_function(u)