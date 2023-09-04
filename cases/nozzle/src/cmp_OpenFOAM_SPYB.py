import meshio #pip3 install h5py meshio
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm

from setup import *
from helpers import meshConvert, findStuff

# Relevant parameters
Re=200000
S=1

# Dimensionalised stuff
r=1.2
O=np.pi/360 # 0.5°
sin,cos=np.sin(O),np.cos(O)
H,L=15,50 # Actual dolfinx mesh size

# Load result of SPYB computations
spyb.loadBaseflow(Re,S)

# Use OpenFOAM mesh
meshConvert("/home/shared/cases/nozzle/mesh/nozzle")
spyb2=SPYB(params,data_path,"nozzle",direction_map)

# Handlers
U,P=spyb2.Q.split()
Nu=spyb2.Nu

# Read OpenFOAM, write mesh
if p0:
	# Searching closest file with respect to setup parameters
	closest_file_name=findStuff("/home/shared/cases/nozzle/baseflow/OpenFOAM/",{'S':S,'Re':Re}, lambda f: f[-3:]=="xmf",False)
	# Read OpenFOAM data
	openfoam_data = meshio.read(closest_file_name)
	print("Loaded "+closest_file_name+" successfully !", flush=True)

	# Read cell_centers
	center_points = openfoam_data.cell_data['CellCenters'][0]
	fine_xy = np.vstack((openfoam_data.points,center_points)) # Regroup all data coordinates
	fine_xy[:,1]/=cos # Plane tilted

	# Reducing problem size (coarse mesh is also smaller)
	msk = (fine_xy[:,0]<r*L)*(fine_xy[:,1]<r*H)
	fine_xy=fine_xy[msk,:2]

	# Dimensionless
	uxv,urv,uthv = np.vstack((openfoam_data.point_data['U'],openfoam_data.cell_data['U'][0]))[msk,:].T
	pv = np.hstack((openfoam_data.point_data['p'],openfoam_data.cell_data['p'][0]))[msk]
	nutv = np.hstack((openfoam_data.point_data['nut'],openfoam_data.cell_data['nut'][0]))[msk]
else: uxv,urv,uthv,pv,nutv,fine_xy=None,None,None,None,None,None

# Data available to all but not distributed
uxv  = comm.bcast(uxv,  root=0)
urv  = comm.bcast(urv,  root=0)
uthv = comm.bcast(uthv, root=0)
pv = comm.bcast(pv, root=0)
nutv = comm.bcast(nutv, root=0)
fine_xy = comm.bcast(fine_xy, root=0)

# Handlers (still useful when !interpolate, uses splines)
def interp(v,x): return griddata(fine_xy,v,x[:2,:].T,'linear')
# Fix orientation
urv,uthv=cos*urv+sin*uthv,-sin*urv+cos*uthv
# Fix no swirl edge case
if S==0: uthv[:]=0

# Map data onto dolfinx vectors
U.sub(0).interpolate(lambda x: interp(uxv, x))
U.sub(1).interpolate(lambda x: interp(urv, x)*(x[1]>params['atol'])) # Enforce u_r=u_th=0 at r=0
U.sub(2).interpolate(lambda x: interp(uthv,x)*(x[1]>params['atol']))
P.interpolate(lambda x: interp(pv,x))
Nu.interpolate(lambda x: interp(nutv,x))

# Total area
A=dfx.fem.assemble_scalar(dfx.fem.form(ufl.dx)).real
A=comm.gather(A)
if p0: A = sum(A)

# Additional handlers
dU,dP=spyb.Q.split()
dNu=spyb.Nu

dU.sub(0).interpolate(dfx.fem.Expression(ufl.sqrt((dU[0]-U[0])**2),spyb.TH0.sub(0).element.interpolation_points()))
dU.sub(1).interpolate(dfx.fem.Expression(ufl.sqrt((dU[1]-U[1])**2),spyb.TH0.sub(0).element.interpolation_points()))
dU.sub(2).interpolate(dfx.fem.Expression(ufl.sqrt((dU[2]-U[2])**2),spyb.TH0.sub(0).element.interpolation_points()))
# Save
spyb.printStuff(spyb.baseflow_path,"dU",dU)
e=dfx.fem.assemble_scalar(dfx.fem.form((dU[0]+dU[1]+dU[2])/3*ufl.dx)).real
e=comm.gather(e)
if p0: print("Average velocity difference:",sum(e)/A)
dP.interpolate(dfx.fem.Expression(ufl.sqrt((dP-P)**2),spyb.TH1.element.interpolation_points()))
# Save
spyb.printStuff(spyb.baseflow_path,"dP",dP)
e=dfx.fem.assemble_scalar(dfx.fem.form(dP*ufl.dx)).real
e=comm.gather(e)
if p0: print("Average pressure difference:",sum(e)/A)
dNu.interpolate(dfx.fem.Expression(ufl.sqrt((dNu-Nu)**2),spyb.TH1.element.interpolation_points()))
# Save
spyb.printStuff(spyb.baseflow_path,"dNu",dNu)
e=dfx.fem.assemble_scalar(dfx.fem.form(dNu*ufl.dx)).real
e=comm.gather(e)
if p0: print("Average eddy viscosity difference:",sum(e)/A)