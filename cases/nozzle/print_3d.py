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

_   = SPY(params,datapath,"baseflow",     direction_map) # Must be first !
spyp=SPYP(params,datapath,"perturbations",direction_map)

# Parameters range
Ss=[0,.2,.4,1]
ms=range(-4,5,2)
Sts=np.linspace(.05,2,5)
# Actual plotting
dir=spyp.resolvent_path+"/3d/"
dirCreator(dir)
p0=comm.rank==0

def frame_args(duration):
    return {"frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},}

# Nozzle surface
y = np.cos(np.linspace(0,np.pi,15))
x,y = np.meshgrid([0,1],y)
z = np.sqrt(1-y**2)
up_nozzle   = go.Surface(x=x, y=y, z= z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")
down_nozzle = go.Surface(x=x, y=y, z=-z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")

# Custom nonuniform grids with maximum resolution at the nozzle
Xf  = np.hstack((np.flip(1-np.geomspace(1e-3,1,30)),np.geomspace(1,1.1,20)))
YZf = np.hstack((np.linspace(0,1,20,endpoint=False),np.geomspace(1,1.3,8))) # Much harder to do something smart about the radius
YZf = np.hstack((np.flip(-YZf)[:-1],YZf))
XYZf = np.meshgrid(Xf,YZf,YZf)
XYZf = np.vstack([C.flatten() for C in XYZf])
#XYZf = XYZf[:,np.maximum(np.abs(XYZf[1]),np.abs(XYZf[2]))>.5] # Cut out central rectangle
#XYZf = XYZf[:,(np.abs(XYZf[1])<1.5)+(np.abs(XYZf[2])<1.5)] # Cut out corners

Xr  = np.hstack((np.flip(1-np.geomspace(1e-3,.8,20)),np.geomspace(1,15,30)))
YZr = np.hstack((np.linspace(0,1,20,endpoint=False), np.geomspace(1,2,10))) # Much harder to do something smart about the radius
YZr = np.hstack((np.flip(-YZr)[:-1],YZr)) # Careful of 0 !
XYZr = np.meshgrid(Xr,YZr,YZr)
XYZr = np.vstack([C.flatten() for C in XYZr])
#XYZr = XYZr[:,(np.abs(XYZr[1])<1.1)+(np.abs(XYZr[2])<1.1)] # Cut out corners

fig = go.Figure(data=[go.Scatter3d(x=XYZf[0],y=XYZf[1],z=XYZf[2], mode='markers', marker={"size":3,"opacity":.4})])
fig.write_html("XYZf.html")
fig = go.Figure(data=[go.Scatter3d(x=XYZr[0],y=XYZr[1],z=XYZr[2], mode='markers', marker={"size":3,"opacity":.4})])
fig.write_html("XYZr.html")

for S in Ss:
	for m in ms:
		for St in Sts:
			if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.2f})",flush=True)
			#try:
			isos_f=spyp.computeIsosurfaces(m,XYZf,.1,spyp.readMode("forcing", Re,S,m,St),6,'Earth',"axial forcing")
			isos_r=spyp.computeIsosurfaces(m,XYZr,.1,spyp.readMode("response", Re,S,m,St),6,'RdBu',"axial response")
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