import meshio, ufl #pip3 install --no-binary=h5py h5py meshio
import numpy as np
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm

# Dimensionalised stuff
R,U_M=.1,10
O=np.pi/360
S,C=np.sin(O),np.cos(O)
if comm.rank==0:
    # Read mesh and point data
    openfoam_data = meshio.read("front.xmf")
    openfoam_data.points[:,:2]/=R*C # Scaling & Plane tilted
    # Write it out again in a dolfinx friendly format
    fine_mesh = meshio.Mesh(points=openfoam_data.points[:,:2],
                            cells={"quad": openfoam_data.get_cells_type("quad")})
    meshio.write("nozzle_fine.xdmf", fine_mesh)
    # Read coarse data (already plane and non tilted)
    dolfinx_data = meshio.read("nozzle_coarse_2D.msh")
    # Write it out again in a dolfinx friendly format
    coarse_mesh = meshio.Mesh(points=dolfinx_data.points[:,:2],
                              cells={"quad": dolfinx_data.get_cells_type("quad")})
    meshio.write("nozzle.xdmf", coarse_mesh)
else: openfoam_data=None
openfoam_data = comm.bcast(openfoam_data, root=0) # data available to all but not distributed
# Read it again in dolfinx - now it's a dolfinx object and it's split amongst procs
with XDMFFile(comm, "nozzle_fine.xdmf", "r") as file:
    fine_mesh = file.read_mesh(name="Grid")
with XDMFFile(comm, "nozzle.xdmf", "r") as file:
    coarse_mesh = file.read_mesh(name="Grid")

# Create FiniteElement, FunctionSpace & Functions
FE_vector  =ufl.VectorElement("Lagrange",coarse_mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("Lagrange",coarse_mesh.ufl_cell(),2,3)
FE_scalar  =ufl.FiniteElement("Lagrange",coarse_mesh.ufl_cell(),1)
V =dfx.FunctionSpace(coarse_mesh, FE_vector)
V2=dfx.FunctionSpace(coarse_mesh, FE_vector_2)
W =dfx.FunctionSpace(coarse_mesh, FE_scalar)
u, u2  = dfx.Function(V), dfx.Function(V2)
p, nut = dfx.Function(W), dfx.Function(W)

# Global indices associated with local ownership
ids_f = fine_mesh.geometry.input_global_indices

# Handlers
def interp(v):
    return griddata(openfoam_data.points[ids_f,:2],v[ids_f],coarse_mesh.geometry.x[:,:2])

# Dimensionless
uxv,urv,uthv = openfoam_data.point_data['U'].T/U_M
pv   = openfoam_data.point_data['p']/U_M**2
nutv = openfoam_data.point_data['nut']/U_M/R
# Fix orientation
urv,uthv=C*urv+S*uthv,-S*urv+C*uthv

# Map data onto dolfinx vectors
u.x.array[:]  =np.hstack((interp(uxv).reshape( (-1,1)),
                          interp(urv).reshape( (-1,1)),
                          interp(uthv).reshape((-1,1)))).flatten()
p.x.array[:]  =interp(pv)
nut.x.array[:]=interp(nutv)
# Interpolation to higher order
u2.interpolate(u)

# Write result as mixed
TH = dfx.FunctionSpace(coarse_mesh,FE_vector_2*FE_scalar)
_, dofs_U = TH.sub(0).collapse(collapsed_dofs=True)
_, dofs_p = TH.sub(1).collapse(collapsed_dofs=True)
q = dfx.Function(TH)
q.x.array[dofs_U]=u2.x.array
q.x.array[dofs_p]=p.x.array

u,p=q.split()
with XDMFFile(comm, "sanity_check_reader.xdmf", "w") as xdmf:
    xdmf.write_mesh(coarse_mesh)
    xdmf.write_function(u)

# Write turbulent viscosity separately
viewer = pet.Viewer().createMPIIO(f"./baseflow/nut/nut_S=0.000_Re=10_n={comm.size}.dat", 'w', comm)
nut.vector.view(viewer)
viewer = pet.Viewer().createMPIIO(f"./baseflow/dat_complex/baseflow_S=0.000_Re=10_n={comm.size}.dat", 'w', comm)
q.vector.view(viewer)