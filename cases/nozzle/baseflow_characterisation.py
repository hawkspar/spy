import meshio #pip3 install --no-binary=h5py h5py meshio
from setup import *
from spy import crl, dirCreator
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm

# Dimensionalised stuff
C = np.cos(np.pi/360) # 0.5°
Q = (1-U_m)/(1+U_m)
p0=comm.rank==0
S=1

openfoam=False
save_str=f"_Re={Re}_S={S:.1f}".replace('.',',')
dir='baseflow/characteristics/'
dirCreator('baseflow')
dirCreator(dir)

# Read OpenFOAM
if openfoam:
	# Read mesh and point data
	openfoam_data = meshio.read("front"+save_str+".xmf")
	xy = openfoam_data.points[:,:2]/C # Scaling & Plane tilted

	# Dimensionless
	ud = openfoam_data.point_data['U'].T

	def interp(v,target_xy,coord=0): return griddata(xy, v[:,coord], target_xy, 'cubic')
# Read dolfinx
else:
	spy = SPY(params, data_path, 'baseflow', direction_map)
	spy.loadBaseflow(Re,S)
	ud,_=spy.Q.split()

	def interp(_, target_xy,coord=0):
		r = spy.eval(ud.split()[coord],target_xy)
		if p0: return r

n = 1000
X = np.linspace(0,50,n)
target_xy = np.zeros((n,2))
target_xy[:,0] = X
u = interp(ud,target_xy)

if p0:
	plt.plot(X, u)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$U_{x,r=0}$')
	plt.savefig(dir+"Ur0(x)"+save_str+".png")
	plt.close()

R = np.linspace(0,10,n)
target_xy = np.ones((n,2))+1e-6
target_xy[:,1] = R
u = interp(ud,target_xy,2)

if p0:
	plt.plot(R, R**2*u**2)
	plt.xlabel(r'$r$')
	plt.ylabel(r'$r^2U_{\theta,x=1+\epsilon}^2$')
	plt.savefig(dir+"r2Ut2"+save_str+".png")
	plt.close()

crls=crl(spy.r,direction_map['x'],direction_map['r'],direction_map['th'],spy.mesh,ud,0)[0]
expr=dfx.fem.Expression(crls.dx(direction_map['r']),spy.TH1.element.interpolation_points())
crls = Function(spy.TH1)
crls.interpolate(expr)
spy.printStuff(dir,"dcrl"+save_str,crls)

X = np.linspace(1,50,n)
RR, XX = np.meshgrid(R,X)
target_xy = np.vstack((XX.flatten(),RR.flatten())).T
u = interp(ud,target_xy)

X_nozzle = np.linspace(0,1,10)
R_nozzle = np.linspace(0,1,n)
RR, XX = np.meshgrid(R_nozzle,X_nozzle)
target_xy = np.vstack((XX.flatten(),RR.flatten())).T
u_nozzle = interp(ud,target_xy)

if p0:
	u = u.reshape((n,n))
	Rc = R.reshape([1,-1])
	u_nozzle = u_nozzle.reshape((10,n))
	Rc_nozzle = R_nozzle.reshape([1,-1])
	min_u=np.tile(np.min(u,1),(n,1)).T
	ths 	   = np.trapz((u-min_u)*(1-u)*Rc,		  R)
	ths_nozzle = np.trapz(u_nozzle*(1-u_nozzle)*Rc_nozzle,R_nozzle)
	plt.plot(np.hstack((X_nozzle,X)), np.hstack((ths_nozzle,ths)))
	plt.xlabel(r'$x$')
	plt.ylabel(r'$\theta$')
	plt.savefig(dir+"theta(x)"+save_str+".png")
	plt.close()

	sgths = np.gradient(ths,X)*U_m/Q

	plt.plot(X, sgths)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$u_{co}/Q d_x\theta$')
	plt.savefig(dir+"Cdxtheta(x)"+save_str+".png")