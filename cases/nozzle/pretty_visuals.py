# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP # Must be after setup
import plotly.graph_objects as go
from mpi4py.MPI import COMM_WORLD as comm

spyp=SPYP(params,datapath,direction_map)
m=3
St=.05
# Actual plotting
dir=spyp.resolvent_path+"/3d/"
spyp.dirCreator(dir)
iso_f=spyp.computeIsosurface(m,0, 1.2, 1.5,100,100,.1,spyp.readCurl("forcing", Re,S,m,St),'Earth')
iso_r=spyp.computeIsosurface(m,.5,16.5,2,  100,100,.1,spyp.readMode("response",Re,S,m,St),'RdBu')
Ri = Function(spyp.TH1)
Ri.interpolate(lambda x: np.isclose(x[1],.99))
nozzle=spyp.computeIsosurface(0,0,1,1,120,120,1,Ri,'Greys')
if comm.rank==0:
	fig=go.Figure(data=[iso_f, iso_r, nozzle])
	fig.update_coloraxes(showscale=False)
	fig.update_layout(scene_aspectmode='data')

	fig.write_html(dir+f"Re={Re:d}_S={S}_m={m:d}_St={St:.2f}".replace('.',',')+".html")