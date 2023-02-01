import numpy as np
import meshio, ufl, sys #pip3 install --no-binary=h5py h5py meshio
from setup import Re, nut, S, params
from dolfinx.io import XDMFFile
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function

sys.path.append('/home/shared/src')

from spy import dirCreator, meshConvert, findStuff, saveStuff

p0=comm.rank==0

real_mode=False
interpolate=True
sanity_check=True
cell_type="triangle"

# Dimensionalised stuff
L,H=36,7
O=np.pi/360 # 0.5°
sin,cos=np.sin(O),np.cos(O)

# Read OpenFOAM, write mesh
if p0:
    # Searching closest file with respect to setup parameters
    closest_file_name=findStuff("./baseflow/OpenFOAM/",['S','Re'],[S,nut], lambda f: f[-3:]=="xmf",False)
    # Read OpenFOAM data
    openfoam_data = meshio.read(closest_file_name)
    print("Loaded "+closest_file_name+" successfully !", flush=True)

    # Read cell_centers
    center_points = openfoam_data.cell_data['CellCenters'][0]
    fine_xy = np.vstack((openfoam_data.points,center_points)) # Regroup all data coordinates
    fine_xy[:,1]/=cos # Plane tilted

    # Reducing problem size (coarse mesh is also smaller)
    msk = (fine_xy[:,0]<L)*(fine_xy[:,1]<H)
    fine_xy=fine_xy[msk,:2]

    # Dimensionless
    uxv,urv,uthv = np.vstack((openfoam_data.point_data['U'],openfoam_data.cell_data['U'][0]))[msk,:].T
    pv   = np.hstack((openfoam_data.point_data['p'],  openfoam_data.cell_data['p'][0])  )[msk]
    nutv = np.hstack((openfoam_data.point_data['nut'],openfoam_data.cell_data['nut'][0]))[msk]

    # Convert mesh
    if convert: meshConvert("perturbations")
else: uxv,urv,uthv,pv,nutv,fine_xy=None,None,None,None,None,None

# Data available to all but not distributed
uxv  = comm.bcast(uxv,  root=0)
urv  = comm.bcast(urv,  root=0)
uthv = comm.bcast(uthv, root=0)
pv   = comm.bcast(pv,   root=0)
nutv = comm.bcast(nutv, root=0)
fine_xy = comm.bcast(fine_xy, root=0)

# Read it again in dolfinx - now it's a dolfinx object and it's split amongst procs
with XDMFFile(comm, "perturbations.xdmf", "r") as file: mesh = file.read_mesh(name="Grid")
if p0: print("Loaded perturbations.xdmf successfully !")

# Create FiniteElement, FunctionSpace & Functions
FE_vector =ufl.VectorElement("CG",mesh.ufl_cell(),2,3)
FE_scalar =ufl.FiniteElement("CG",mesh.ufl_cell(),1)
FE_scalar2=ufl.FiniteElement("CG",mesh.ufl_cell(),2)
V=FunctionSpace(mesh, FE_vector)
W=FunctionSpace(mesh, FE_scalar)
X=FunctionSpace(mesh, FE_scalar2)
U, P, Nu = Function(V), Function(W), Function(X)

# Handlers (still useful when !interpolate)
def interp(v,x): return griddata(fine_xy,v,x[:2,:].T,'cubic')
# Fix orientation
urv,uthv=cos*urv+sin*uthv,-sin*urv+cos*uthv
# Fix no swirl edge case
if S==0: uthv[:]=0

# Map data onto dolfinx vectors
U.sub(0).interpolate(lambda x: interp(uxv, x))
U.sub(1).interpolate(lambda x: interp(urv, x)*(x[1]>params['atol'])) # Enforce u_r=u_th=0 at r=0
U.sub(2).interpolate(lambda x: interp(uthv,x)*(x[1]>params['atol']))
P.interpolate( lambda x: interp(pv,  x))
Nu.interpolate(lambda x: interp(nutv,x))
# Fix negative eddy viscosity
Nu.x.array[Nu.x.array<0] = 0

# Save pretty graphs
if sanity_check:
    for f in ['U','P','Nu']:
        with XDMFFile(comm, "sanity_check_"+f+"_reader.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            eval("xdmf.write_function("+f+")")

# Write all functions separately
pre="./baseflow"
dirCreator(pre)

if type(S)==int: app=f"_S={S:d}_Re={nut:d}_nut={nut:d}"
else:            app=f"_S={S:.1f}_Re={nut:d}_nut={nut:d}".replace('.',',')

# Save
saveStuff(pre+"/u/",  "u"  +app,U)
saveStuff(pre+"/p/",  "p"  +app,P)
saveStuff(pre+"/nut/","nut"+app,Nu)