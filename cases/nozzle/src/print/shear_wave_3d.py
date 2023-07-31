# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from grids_3d import *
from spyp import SPYP # Must be after setup
from helpers import dirCreator

spyp = SPYP(params,data_path,pert_mesh,direction_map)

# Actual plotting
dir=spyp.resolvent_path+"3d/"
dirCreator(dir)

directions=list(direction_map.keys())

print_list=[#{'S':1,'m':-2,'St':7.3057e-03/2,'XYZ':XYZc_es},
			{'S':1,'m': 2,'St':7.3057e-03/2,'XYZ':XYZq_es},
			#{'S':1,'m': 2,'St':0,		    'XYZ':XYZc_st},
			#{'S':1,'m':-2,'St':0,		    'XYZ':XYZc_sw}
]
S_save=-1

for dat in print_list:
	S,m,St,XYZ=dat['S'],dat['m'],dat['St'],dat['XYZ']
	if S_save!=S:
		spyb.loadBaseflow(Re,S,False)
		S_save=S
	small_dat={'Re':Re,'S':S,'m':m,'St':St}
	if p0: print(f"Currently quiverying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.2f})",flush=True)
	file_name=dir+f"shear_wave_Re={Re:d}_S={S:.1f}_m={m:d}_St={St:.4e}".replace('.',',')+".html" # Usual Re & St based on D
	"""if isfile(file_name):
		if p0: print("Found html file, moving on...",flush=True)
		continue"""
	cones_w=spyp.computeWavevector("response",small_dat,XYZ,1,1,'Picnic',"response")
	cone_s =spyb.computeShear(XYZ,"Greens")
	# Plus d'animation - on économise la place !
	if p0:
		fig = go.Figure(data=[up_nozzle, down_nozzle, cones_w[0], cone_s])
		fig.update_coloraxes(showscale=False)
		fig.update_layout(scene_aspectmode='data')
		fig.write_html(file_name)
		print(f"Wrote {file_name}",flush=True)