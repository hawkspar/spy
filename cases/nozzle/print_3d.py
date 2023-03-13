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

Ss=[.2,.4,0,1]
ms=range(-5,6)
Sts=[.05,.25]
# Actual plotting
dir=spyp.resolvent_path+"/3d/"
dirCreator(dir)
p0=comm.rank==0

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

for S in Ss:
	for m in ms:
		for St in Sts:
			if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S},{m},{St})",flush=True)
			try:
				iso_f=spyp.computeIsosurface(m,0, 1.5, 1.5,200,200,.1,spyp.readMode("forcing", Re,S,m,St),'Earth')
				iso_r=spyp.computeIsosurface(m,0,10,     2,200,200,.1,spyp.readMode("response",Re,S,m,St),'RdBu')
			except (FileNotFoundError, ValueError): continue
			x, y = np.mgrid[0:1:100j, -1:1:100j]
			z = np.sqrt(1-y**2)
			up_nozzle   = go.Surface(x=x, y=y, z= z, colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(0,0,0)']],opacity=.5)
			down_nozzle = go.Surface(x=x, y=y, z=-z, colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(0,0,0)']],opacity=.5)
			if p0:
				fig=go.Figure(data=[iso_f, iso_r, up_nozzle, down_nozzle])

				fig = go.Figure(
					data=[isos_f[0], isos_r[0], up_nozzle, down_nozzle],
					layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
					frames=[go.Frame(data=[isos_f[i], isos_r[i], up_nozzle, down_nozzle], name=str(i)) for i in range(len(isos_f))]
				)
				sliders = [
							{
								"pad": {"b": 10, "t": 60},
								"len": 0.9,
								"x": 0.1,
								"y": 0,
								"steps": [
									{
										"args": [[f.name], frame_args(0)],
										"label": str(k),
										"method": "animate",
									}
									for k, f in enumerate(fig.frames)
								],
							}
						]
					
				fig.update_layout(sliders=sliders)
				fig.update_coloraxes(showscale=False)
				fig.update_layout(scene_aspectmode='data')

				fig.write_html(dir+f"Re={Re:d}_S={S}_m={m:d}_St={St:.2f}".replace('.',',')+".html")
