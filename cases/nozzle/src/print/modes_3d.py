# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path
from os.path import isfile
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default="firefox"

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from grids_3d import *
from spyp import SPYP # Must be after setup
from helpers import dirCreator

spyp = SPYP(params,data_path,pert_mesh,direction_map)

# Actual plotting
dir=spyp.resolvent_path+"/3d/"
dirCreator(dir)

Ss_ref = [1]
ms_ref = [-2]
Sts_ref = [.00730566]

# Nozzle surface
y = np.cos(np.linspace(0,np.pi,15))
x,y = np.meshgrid([0,1],y)
z = np.sqrt(1-y**2)
up_nozzle   = go.Surface(x=x, y=y, z= z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")
down_nozzle = go.Surface(x=x, y=y, z=-z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")

def frameArgs(duration):
    return {"frame": {"duration": duration}, "mode": "immediate", "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},}

if p0:
	fig = go.Figure(data=[go.Scatter3d(x=XYZf_sk[0],y=XYZf_sk[1],z=XYZf_sk[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZf_sk.html")
	fig = go.Figure(data=[go.Scatter3d(x=XYZr_sk[0],y=XYZr_sk[1],z=XYZr_sk[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZr_sk.html")
	fig = go.Figure(data=[go.Scatter3d(x=XYZf_kh[0],y=XYZf_kh[1],z=XYZf_kh[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZf_kh.html")
	fig = go.Figure(data=[go.Scatter3d(x=XYZr_kh[0],y=XYZr_kh[1],z=XYZr_kh[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZr_kh.html")

for S in Ss_ref:
	for m in ms_ref:
		for St in Sts_ref:
			file_name=dir+f"Re={Re:d}_S={S:.1f}_m={m:d}_St={St:.4e}".replace('.',',') # Usual Re & St based on D but who cares
			dat={"Re":Re,"S":S,"m":m,"St":St}
			if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.4e})",flush=True)
			"""if isfile(file_name):
				if p0: print("Found an html file, moving on...",flush=True)
				continue
			if St > .5 or m == 0:
				isos_f=spyp.computeIsosurfaces("forcing", dat,XYZf_kh,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_kh,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[0] and S == 0:
				isos_f=spyp.computeIsosurfaces("forcing", dat,XYZf_sk,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_sh,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[0] and S < .5:
				isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_sk,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_sk,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[0] and m > 0:
				isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_sk,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_st,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[0] and m < 0:
				isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_sk,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_sw,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[-1]:
				isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_kh,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_kh,.1,2,'Picnic',"axial response")
			else:
				isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_cr,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_nr,.1,2,'Picnic',"axial response")"""
			#isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_es,.1,2,'Earth', "axial forcing")
			isos_r=spyp.computeIsosurfaces("response",dat,XYZr_sw,.1,1,'Picnic',"axial response")
			# Animation
			if p0:
				for i,d in enumerate(['x']):
					"""fig = go.Figure(data=[#isos_f[i][0],
										isos_r[i][0], up_nozzle, down_nozzle],
									layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
									frames=[go.Frame(data=[#isos_f[i][j],
									isos_r[i][j], up_nozzle, down_nozzle], name=str(j)) for j in range(len(isos_r))])
					fig.update_layout(sliders=[{"pad": {"b": 10, "t": 60}, "len": .9, "x": .1, "y": 0,
												"steps": [{"args": [[f.name], frameArgs(0)], "label": str(k),
														"method": "animate",} for k, f in enumerate(fig.frames)],}]	)"""
					fig = go.Figure(data=[#isos_f[i][0],
										  isos_r[i][0], up_nozzle, down_nozzle])
					fig.update_coloraxes(showscale=False)
					fig.update_layout(scene_aspectmode='data')
					fig.write_html(file_name+'_dir='+d+".html")
					if p0: print(f"Wrote {file_name}_dir={d}.html",flush=True)