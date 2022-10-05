import re, os
import numpy as np
import PetscBinaryIO #export PYTHONPATH=/usr/local/petsc/lib/petsc/bin/:$PYTHONPATH
import meshio, ufl, sys #pip3 install --no-binary=h5py h5py meshio
from setup import Re, S, params
from dolfinx.io import XDMFFile
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function

sys.path.append('/home/shared/src')

from spy import dirCreator, meshConvert

p0=comm.rank==0

real_mode=False
interpolate=True
sanity_check=False
cell_type="triangle"

# Dimensionalised stuff
R,U_M=.1,10
L,H=36,5
O=np.pi/360 # 0.5°
sin,cos=np.sin(O),np.cos(O)

# Read OpenFOAM, write mesh
if p0:
    # Searching closest file with respect to setup parameters
    file_names = [f for f in os.listdir(".") if f[-3:]=="xmf"]
    d=np.infty
    for file_name in file_names:
        fd=0
        for param,key in zip([Re,S],["Re","S"]):
            match = re.search(r'_'+key+r'=(\d*\.?\d*)',file_name)
            param_file = float(match.group(1)) # Take advantage of file format
            fd += abs(param-param_file)
        if fd<d: d,closest_file_name=fd,file_name
    
    # Read OpenFOAM data
    openfoam_data = meshio.read(closest_file_name)
    print("Loaded "+closest_file_name+" successfully !", flush=True)
    openfoam_data.points[:,:2]/=R # Scaling & Plane tilted
    openfoam_data.points[:, 1]/=cos

    # Convert mesh
    if interpolate:
        # Write it out again in a dolfinx friendly format
        meshConvert("nozzle_2D_coarse","nozzle",cell_type)
    else:
        # Important to ensure consistancy in partioning
        meshConvert("nozzle_2D","nozzle",cell_type)
else: openfoam_data=None

openfoam_data = comm.bcast(openfoam_data, root=0) # data available to all but not distributed
# Read it again in dolfinx - now it's a dolfinx object and it's split amongst procs
with XDMFFile(comm, "nozzle.xdmf", "r") as file:
    mesh = file.read_mesh(name="Grid")

# Create FiniteElement, FunctionSpace & Functions
FE_vector  =ufl.VectorElement("CG",mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("CG",mesh.ufl_cell(),2,3)
FE_scalar  =ufl.FiniteElement("CG",mesh.ufl_cell(),1)
V =FunctionSpace(mesh, FE_vector)
V2=FunctionSpace(mesh, FE_vector_2)
W =FunctionSpace(mesh, FE_scalar)
u, U   = Function(V), Function(V2)
P, nut = Function(W), Function(W)

# Handlers (still useful when !interpolate)
fine_xy=openfoam_data.points[:,:2]
coarse_xy=mesh.geometry.x[:,:2]

def interp(v,reshape=False):
    v=griddata(fine_xy,v,coarse_xy,'cubic')
    if reshape: return v.reshape((-1,1))
    return v

# Reducing problem size (coarse mesh is also smaller)
msk = np.logical_and(fine_xy[:,0]<L,fine_xy[:,1]<H)
fine_xy=fine_xy[msk]

# Dimensionless
uxv,urv,uthv = (openfoam_data.point_data['U'].T)[:,msk]/U_M
pv   = openfoam_data.point_data['p'][msk]/U_M**2
nutv = openfoam_data.point_data['nut'][msk]/U_M/R
# Fix orientation
urv,uthv=cos*urv+sin*uthv,-sin*urv+cos*uthv
# Fix no swirl edge case
if S==0: uthv[:]=0
# Fix negative eddy visocisty
nutv[nutv<0] = 0

# Map data onto dolfinx vectors
u.x.array[:]=np.hstack((interp(uxv, 1),
                        interp(urv, 1),
                        interp(uthv,1))).flatten()
P.x.array[:]  =interp(pv, 0)
nut.x.array[:]=interp(nutv, 0)

# BAD because interpolation in wrong order
U.interpolate(u)

# Save pretty graphs
if sanity_check:
    for f in ['U','P','nut']:
        with XDMFFile(comm, "sanity_check_"+f+"_reader.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            eval("xdmf.write_function("+f+")")

# Write all functions separately
pre="./baseflow"
typ=real_mode*"real"+(1-real_mode)*"complex"
dirCreator(pre)
dirCreator(pre+"/u/")
dirCreator(pre+"/p/")
dirCreator(pre+"/nut/")
dirCreator(pre+"/u/"  +typ)
dirCreator(pre+"/p/"  +typ)
dirCreator(pre+"/nut/"+typ)

app=f"_S={S:.3f}_Re={Re:d}_n={comm.size:d}_p={comm.rank:d}.dat"

# Binary IO ; parallel
io = PetscBinaryIO.PetscBinaryIO(complexscalars=not real_mode)
U_vec = U.vector.array_w.view(PetscBinaryIO.Vec)
P_vec = P.vector.array_w.view(PetscBinaryIO.Vec)
nut_vec = nut.vector.array_w.view(PetscBinaryIO.Vec)
comm.barrier() # Important otherwise file corrupted

io.writeBinaryFile(pre+"/u/"  +typ+"/u"  +app, [U_vec])
io.writeBinaryFile(pre+"/p/"  +typ+"/p"  +app, [P_vec])
io.writeBinaryFile(pre+"/nut/"+typ+"/nut"+app, [nut_vec])