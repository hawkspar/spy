# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP # Must be after setup
from spy import dirCreator
import plotly.graph_objects as go
from mpi4py.MPI import COMM_WORLD as comm

_    =  SPY(params,datapath,"baseflow",     direction_map) # Must be first !
spyp = SPYP(params,datapath,"perturbations",direction_map)

# Parameters range
Ss=[0]#[0,.2,.4,1]
ms=[0]#range(-4,5,2)
Sts=[1.02]#np.linspace(.05,2,5)
# Actual plotting
dir=spyp.resolvent_path+"/3d/"
dirCreator(dir)
p0=comm.rank==0

def frame_args(duration):
    return {"frame": {"duration": duration},
            "mode": "immediate", "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},}

# Nozzle surface
y = np.cos(np.linspace(0,np.pi,15))
x,y = np.meshgrid([0,1],y)
z = np.sqrt(1-y**2)
up_nozzle   = go.Surface(x=x, y=y, z= z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")
down_nozzle = go.Surface(x=x, y=y, z=-z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")

# Custom nonuniform grids
# Streaks : long elongated structures, maximum resolution at the nozzle
Xf_sk  = np.hstack((np.flip(1-np.geomspace(.0001,1,35)),np.linspace(1,1.3,10)))
YZf_sk = np.hstack((np.linspace(0,1,25,endpoint=False), np.linspace(1,1.3,25)))
YZf_sk = np.hstack((np.flip(-YZf_sk)[:-1],YZf_sk)) # Careful of 0 !
XYZf_sk = np.meshgrid(Xf_sk,YZf_sk,YZf_sk)
XYZf_sk = np.vstack([C.flatten() for C in XYZf_sk])
def box_f_sk(x): return (x[:,1]**2+x[:,2]**2>.75**2)*(x[:,1]**2+x[:,2]**2<1.1**2)

Xr_sk  = np.hstack((np.linspace(.5,1.1,20,endpoint=False),np.geomspace(1.1,15,30)))
YZr_sk = np.hstack((np.linspace(0,1,20,endpoint=False), np.geomspace(1,2,30)))
YZr_sk = np.hstack((np.flip(-YZr_sk)[:-1],YZr_sk)) # Careful of 0 !
XYZr_sk = np.meshgrid(Xr_sk,YZr_sk,YZr_sk)
XYZr_sk = np.vstack([C.flatten() for C in XYZr_sk])
def box_r_sk(x): return (x[:,1]**2+x[:,2]**2>.5**2)+(x[:,1]**2+x[:,2]**2<2**2)+(x[:,1]**2+x[:,2]**2<1.3**2)*(x[:,0]<1)

# KH : shorter structures nozzle
Xf_kh  = np.linspace(0,1.3,50)
YZf_kh = np.linspace(0,1.3,50)
YZf_kh = np.hstack((np.flip(-YZf_kh)[:-1],YZf_kh)) # Careful of 0 !
XYZf_kh = np.meshgrid(Xf_kh,YZf_kh,YZf_kh)
XYZf_kh = np.vstack([C.flatten() for C in XYZf_kh])
def box_f_kh(x): return x[:,1]**2+x[:,2]**2<1.3**2

Xr_kh  = np.linspace(.9,12,100)
YZr_kh = np.linspace(0,1.2,100)
YZr_kh = np.hstack((np.flip(-YZr_kh)[:-1],YZr_kh)) # Careful of 0 !
XYZr_kh = np.meshgrid(Xr_kh,YZr_kh,YZr_kh)
XYZr_kh = np.vstack([C.flatten() for C in XYZr_kh])
def box_r_kh(x): return x[:,1]**2+x[:,2]**2<1.2**2

Xr_kh2  = np.linspace(.9,8,100)
XYZr_kh2 = np.meshgrid(Xr_kh2,YZr_kh,YZr_kh)
XYZr_kh2 = np.vstack([C.flatten() for C in XYZr_kh2])

fig = go.Figure(data=[go.Scatter3d(x=XYZf_sk[0],y=XYZf_sk[1],z=XYZf_sk[2], mode='markers', marker={"size":3,"opacity":.4})])
fig.write_html("XYZf_sk.html")
fig = go.Figure(data=[go.Scatter3d(x=XYZr_sk[0],y=XYZr_sk[1],z=XYZr_sk[2], mode='markers', marker={"size":3,"opacity":.4})])
fig.write_html("XYZr_sk.html")
fig = go.Figure(data=[go.Scatter3d(x=XYZf_kh[0],y=XYZf_kh[1],z=XYZf_kh[2], mode='markers', marker={"size":3,"opacity":.4})])
fig.write_html("XYZf_kh.html")
fig = go.Figure(data=[go.Scatter3d(x=XYZr_kh[0],y=XYZr_kh[1],z=XYZr_kh[2], mode='markers', marker={"size":3,"opacity":.4})])
fig.write_html("XYZr_kh.html")

for S in Ss:
	for m in ms:
		for St in Sts:
			if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.2f})",flush=True)
			"""try:
				if St == Sts[0]:
					isos_f=spyp.computeIsosurfaces(m,XYZf_sk,.1,spyp.readMode("forcing", Re,S,m,St),1,'Earth',"axial forcing", box_f_sk)
					isos_r=spyp.computeIsosurfaces(m,XYZr_sk,.1,spyp.readMode("response",Re,S,m,St),1,'RdBu', "axial response",box_r_sk)
				elif St == Sts[-1]:
					isos_f=spyp.computeIsosurfaces(m,XYZf_kh, .1,spyp.readMode("forcing", Re,S,m,St),1,'Earth',"axial forcing", box_f_kh)
					isos_r=spyp.computeIsosurfaces(m,XYZr_kh2,.1,spyp.readMode("response",Re,S,m,St),1,'RdBu', "axial response",box_r_kh)
				else:"""
			isos_f=spyp.computeIsosurfaces(m,XYZf_kh,.1,spyp.readMode("forcing", Re,S,m,St),1,'Earth',"axial forcing", box_f_kh)
			isos_r=spyp.computeIsosurfaces(m,XYZr_kh,.1,spyp.readMode("response",Re,S,m,St),1,'RdBu', "axial response",box_r_kh)
			"""except ValueError:
				if p0: print("There was a problem with the modes, moving on...",flush=True)
				continue"""
			# Animation
			if p0:
				fig = go.Figure(data=[isos_f[0], isos_r[0], up_nozzle, down_nozzle],
								layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
								frames=[go.Frame(data=[isos_f[i], isos_r[i], up_nozzle, down_nozzle], name=str(i)) for i in range(len(isos_r))])
				fig.update_layout(sliders=[{"pad": {"b": 10, "t": 60},
											"len": .9, "x": .1, "y": 0,
											"steps": [{"args": [[f.name], frame_args(0)],
													"label": str(k),
													"method": "animate",} for k, f in enumerate(fig.frames)],}]	)
				fig.update_coloraxes(showscale=False)
				fig.update_layout(scene_aspectmode='data')
				fig.write_html(dir+f"Re={Re:d}_S={S}_m={m:d}_St={St:.2f}".replace('.',',')+".html")