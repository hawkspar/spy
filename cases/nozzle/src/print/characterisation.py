from sys import path
from meshio import read #pip3 install --no-binary=h5py h5py meshio
from scipy.stats import linregress
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spy import grd, crl, dirCreator

# Dimensionalised stuff
C = np.cos(np.pi/360) # 0.5°
Q = (1-U_m)/(1+U_m)
S=1

openfoam=False
save_str=f"_Re={Re}_S={S:.1f}".replace('.',',')
dir='/home/shared/cases/nozzle/baseflow/'
dirCreator(dir)
dir+='characteristics/'
dirCreator(dir)

# Read OpenFOAM
if openfoam:
	# Read mesh and point data
	openfoam_data = read("front"+save_str+".xmf")
	xy = openfoam_data.points[:,:2]/C # Scaling & Plane tilted

	# Dimensionless
	ud = openfoam_data.point_data['U'].T

	# Handler to interpolate baseflow
	def interp(v,target_xy,coord=0): return griddata(xy, v[:,coord], target_xy, 'cubic')
# Read dolfinx
else:
	spyb.loadBaseflow(Re,S)
	ud,_=spyb.Q.split()

	# Handler to interpolate baseflow
	def interp(_, target_xy,coord=0):
		r = spyb.eval(ud.split()[coord],target_xy)
		if p0: return r

n = 1000
# Speed on the axis
X = np.linspace(0,50,n)
target_xy = np.zeros((n,2))
target_xy[:,0] = X
u = interp(ud,target_xy)
if p0:
	plt.plot(X, u)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$U_x(x,0)$')
	plt.savefig(dir+"Ur0(x)"+save_str+".png")
	plt.close()

# (ru_th)^2 at nozzle
R = np.linspace(0,15,n)
target_xy = np.ones((n,2))+1e-6
target_xy[:,1] = R
u = interp(ud,target_xy,2)
if p0:
	plt.plot(R, R**2*u**2)
	plt.xlabel(r'$r$')
	plt.ylabel(r'$r^2U_{\theta,x=1+\epsilon}^2$')
	plt.savefig(dir+"r2Ut2"+save_str+".png")
	plt.close()

v_ang = Function(spyb.TH1)
v_ang.interpolate(dfx.fem.Expression(ud[2].dx(direction_map['r'])/spyb.r-ud[2]/spyb.r**2,spyb.TH1.element.interpolation_points()))
spyb.printStuff(dir,"v_ang"+save_str,v_ang)

FS = dfx.fem.FunctionSpace(spyb.mesh,ufl.TensorElement("CG",spyb.mesh.ufl_cell(),2,(3,3)))
grds = Function(FS)
crls = Function(spyb.TH0)

grds_ufl=grd(spyb.r,direction_map['x'],direction_map['r'],direction_map['theta'],ud,0)
crls_ufl=crl(spyb.r,direction_map['x'],direction_map['r'],direction_map['theta'],spyb.mesh,ud,0)
# Baseflow gradient
grds.interpolate(dfx.fem.Expression(grds_ufl,FS.element.interpolation_points()))
spyb.printStuff(dir,"grd"+save_str,grds)
# Baseflow vorticity
crls.interpolate(dfx.fem.Expression(crls_ufl,spyb.TH0.element.interpolation_points()))
spyb.printStuff(dir,"crl"+save_str,crls)
# Derivative of vorticity
crls.interpolate(dfx.fem.Expression(crls_ufl.dx(direction_map['r']),spyb.TH0.element.interpolation_points()))
spyb.printStuff(dir,"d_r crl"+save_str,crls)

# Shear layer thickness
RR, XX = np.meshgrid(R,X)
target_xy = np.vstack((XX.flatten(),RR.flatten())).T
u = interp(ud,target_xy)
if p0:
	ths = np.empty(n)
	for i in range(n):
		u_x = u[i*n:(i+1)*n]
		i_m = np.argmin(u_x)
		v = (u_x[:i_m+1]-u_x[i_m])/(1-u_x[i_m])
		ths[i] = np.trapz(v*(1-v)*R[:i_m+1],R[:i_m+1]) # Again dimensionless with R or D changes things
	print(ths[0])

	m,M=4,10
	msk=(m<X)*(X<M)
	res=linregress(X[msk],ths[msk])
	a,b=res.slope,res.intercept

	dat=np.array([[0, 0.05049088359046272],
				[0.9089110750777021, 0.06303770049416868],
				[1.5267607997818011, 0.06853831977600766],
				[2.1446105244859, 0.07442494742850214],
				[2.7624602491899974, 0.09276034503463237],
				[3.3803099738940965, 0.11013072171412408],
				#[3.2118055035202513, 0.7763811694652958],
				[3.9981596985981938, 0.1284661193202542],
				[4.616009423302293, 0.14680151692638443],
				[5.23385914800639, 0.16455790197653153],
				[5.851708872710489, 0.18144576819270397],
				[6.469558597414586, 0.19746511557490198],
				[7.087408322118685, 0.21280894830845298],
				[7.705258046822783, 0.22747726639335708],
				[8.32310777152688, 0.24233858866358893],
				[8.94095749623098, 0.25642789419251],
				[9.558807220935078, 0.27061370181409483],
				[10.176656945639175, 0.28528201989899904],
				#[10.738338513551993, 0.08356047886734608],
				[10.794506670343273, 0.2997573337985755],
				[11.412356395047373, 0.31365363514216893],
				[12.03020611975147, 0.3281289490417454],
				[12.648055844455568, 0.34327977758996864],
				[13.265905569159669, 0.35891311660151126],
				[13.883755293863766, 0.37560797863235607],
				#[13.827587137072483, 0.08426816088021427],
				[14.501605018567863, 0.39104831345857094],
				[15.11945474327196, 0.4062956440994582],
				#[15.287959213645808, 0.10832934931773242],
				[15.737304467976061, 0.42376252287161376],
				[16.35515419268016, 0.4407468911804501],
				[16.973003917384254, 0.4576347573966226],
				[17.590853642088355, 0.47394361105681204],
				[18.208703366792456, 0.49034896680966533],
				#[18.489544150748863, 0.07789902276440053],
				[18.82655309149655, 0.5063683141918632],
				[19.44440281620065, 0.5242212013346742],
				[20.062252540904744, 0.5433286156821151],
				[20.680102265608845, 0.5619535195662367],
				[21.297951990312946, 0.5797099046163839],
				[21.91580171501704, 0.598720816871161],
				[22.53365143972114, 0.6185037458672487],
				[23.15150116442524, 0.6385761811413281],
				[23.769350889129335, 0.6598066415273736],
				[24.387200613833436, 0.6801685830794444],
				[25.00505033853753, 0.6985039806855746],
				[25.62290006324163, 0.7137513113264617],
				[26.24074978794573, 0.7289986419673489],
				[26.858599512649825, 0.7455970019055299],
				[27.336028845375722, 0.7565660731049867],
				[27.977736549165122, 0.7727910238429171]])

	plt.plot(X, ths, label=r'Present case')#label=r'$\theta=\int_0^{r_0}u(1-u)rdr$')
	plt.plot(dat[:,0]*2, dat[:,1]*4, label=r'Schmidt') # Again dimensionless with R or D changes things
	#plt.plot((0,np.max(X)/2), b+a*np.array((0,np.max(X)))/2,label=r'$y='+f'{a:.3f}x{b:+.3f}$')
	plt.legend()
	plt.xlabel(r'$x$')
	plt.ylabel(r'$\Theta$')
	plt.savefig(dir+"theta(x)"+save_str+".png")
	plt.close()

	sgths = np.gradient(ths,X)*U_m/Q

	plt.plot(X, sgths)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$u_{co}/Q d_x\theta$')
	plt.savefig(dir+"Cdxtheta(x)"+save_str+".png")